// ==================
// Tests operators 3D
// ==================
// Copyright (C) Adeeb Arif Kor

#include "forms.h"
#include "spectral_op.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

using T = double;

using namespace dolfinx;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set polynomial degree
    const int P = 4;

    // Create mesh and function space
    const std::size_t N = 32;
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh<T>>(
      mesh::create_box(
        MPI_COMM_WORLD,
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
        {N, N, N},
        mesh::CellType::hexahedron,
        part));

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_m, "u", mesh));

    auto dofs = V->dofmap()->index_map->size_global();

    if (mpi_rank == 0)
      std::cout << "Number of degrees-of-freedom: " << dofs << "\n";

    // Get index map and block size
    auto index_map = V->dofmap()->index_map;

    // Create input function
    auto u = std::make_shared<fem::Function<T>>(V);
    u->interpolate([](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
      std::vector<T> u(x.extent(1));

      for (std::size_t p = 0; p < x.extent(1); ++p)
        u[p] = std::sin(x(0, p)) * std::cos(std::numbers::pi * x(1, p));

      return {u, {u.size()}};
    });

    // Create DG functions
    auto V_DG = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_m, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();

    std::fill(c0_.begin(), c0_.end(), 1.0);

    // ------------------------------------------------------------------------
    // Mass coefficients
    std::vector<T> m_coeffs(c0_.size());
    for (std::size_t i = 0; i < m_coeffs.size(); ++i)
      m_coeffs[i] = c0_[i];

    // ------------------------------------------------------------------------
    // Mass operator
    MassSpectral3D<T, P> mass_spectral(V);
    auto m1 = std::make_shared<fem::Function<T>>(V);
    auto m1_ = m1->x()->mutable_array();

    common::Timer timing_mass_operator("~ Mass operator timings");

    // Timing mass operator function
    for (int i = 0; i < 10; ++i) {
      std::fill(m1_.begin(), m1_.end(), 0.0);
      timing_mass_operator.start();
      mass_spectral(*u->x(), m_coeffs, *m1->x());
      timing_mass_operator.stop();
      m1->x()->scatter_rev(std::plus<T>());
    }

    // ------------------------------------------------------------------------
    // Stiffness coefficients
    std::vector<T> s_coeffs(c0_.size());
    for (std::size_t i = 0; i < s_coeffs.size(); ++i)
      s_coeffs[i] = c0_[i];

    // ------------------------------------------------------------------------
    // Stiffness operator
    StiffnessSpectral3D<T, P> stiffness_spectral(V);
    auto s1 = std::make_shared<fem::Function<T>>(V);
    auto s1_ = s1->x()->mutable_array();

    common::Timer timing_stiff_operator("~ Stiffness operator timings");

    // Timing stiffness operator function
    for (int i = 0; i < 10; ++i) {
      std::fill(s1_.begin(), s1_.end(), 0.0);
      timing_stiff_operator.start();
      stiffness_spectral(*u->x(), s_coeffs, *s1->x());
      timing_stiff_operator.stop();
      s1->x()->scatter_rev(std::plus<T>());
    }

    // ------------------------------------------------------------------------
    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::average);

  }
  PetscFinalize();
}