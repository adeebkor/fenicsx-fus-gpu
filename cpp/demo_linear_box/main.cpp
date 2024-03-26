//
// Linear wave
// - plane wave
// - homogenous media
// ==================
// Copyright (C) Adeeb Arif Kor

#include "Linear.hpp"
#include "forms.h"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iomanip>
#include <iostream>

#define T_MPI MPI_FLOAT
using T = float;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const T sourceFrequency = 0.5e6;      // (Hz)
    const T sourceAmplitude = 60000;      // (Pa)
    const T period = 1 / sourceFrequency; // (s)

    // Material parameters
    const T speedOfSound = 1500; // (m/s)
    const T density = 1000;      // (kg/m^3)

    // Domain parameters
    const T domainLength = 0.03; // (m)

    // FE parameters
    const int degreeOfBasis = 4;

    // Mesh parameters
    const T waveLength = speedOfSound / sourceFrequency;
    const T numOfWaves = domainLength / waveLength;
    const std::size_t numOfElement = 2 * numOfWaves;

    std::cout << "Wavelength: " << waveLength << "\n";
    std::cout << "Number of waves: " << numOfWaves << "\n";
    std::cout << "Number of element in each coordinate direction: " << numOfElement << "\n";

    // Create mesh
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        mesh::create_box<T>(
            MPI_COMM_WORLD,
            {{{0.0, 0.0, 0.0}, {domainLength, domainLength, domainLength}}},
            {numOfElement, numOfElement, numOfElement},
            mesh::CellType::hexahedron,
            part
        )
    );
    mesh->topology()->create_connectivity(1, 2);

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_a, "u", mesh));

    auto dofs = V->dofmap()->index_map->size_global();

    if (mpi_rank == 0)
      std::cout << "Number of degrees-of-freedom: " << dofs << "\n";

    // Mesh data
    const int tdim = mesh->topology()->dim();
    const std::size_t num_cell = mesh->topology()->index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<T> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<T>::iterator min_mesh_size_local
        = std::min_element(mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx = std::distance(mesh_size_local.begin(), min_mesh_size_local);
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Temporal parameters
    const T CFL = 0.65;
    T timeStepSize = CFL * meshSizeMinGlobal / (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime = domainLength / speedOfSound + 2.0 / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0) {
        std::cout << "Number of steps: " << numberOfStep << "\n";
    }

    // Define DG functions for material parameters
    auto V_DG = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::fill(c0_.begin(), c0_.end(), speedOfSound);

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::fill(rho0_.begin(), rho0_.end(), density);

    // Boundary facet (source)
    auto boundary_facets = mesh::locate_entities_boundary(
        *mesh, tdim-1, 
        [](auto x)
        {
            using U = typename decltype(x)::value_type;
            constexpr U eps = 1.0e-8;
            std::vector<std::int8_t> marker(x.extent(1), false);
            for (std::size_t p = 0; p < x.extent(1); ++p)
            {
                auto x0 = x(0, p);
                if (std::abs(x0) < eps)
                    marker[p] = true;
            }
            return marker;
        });
    std::vector<std::int32_t> tags(boundary_facets.size(), 1);

    // Boundary facet (absorbing)
    auto boundary_facets2 = mesh::locate_entities_boundary(
        *mesh, tdim-1,
        [domainLength](auto x)
        {
            using U = typename decltype(x)::value_type;
            constexpr U eps = 1.0e-8;
            std::vector<std::int8_t> marker(x.extent(1), false);
            for (std::size_t p = 0; p < x.extent(1); ++p)
            {
                auto x0 = x(0, p);
                if (std::abs(x0) > domainLength - eps)
                    marker[p] = true;
            }
            return marker;
        });
    std::vector<std::int32_t> tags2(boundary_facets2.size(), 2);

    boundary_facets.insert(std::end(boundary_facets), std::begin(boundary_facets2), std::end(boundary_facets2));
    tags.insert(std::end(tags), std::begin(tags2), std::end(tags2));

    // Sort boundary facets
    std::vector<std::int32_t> boundary_facets_sorted(boundary_facets.size());
    std::vector<std::int32_t> tags_sorted(tags.size());

    std::vector<std::int32_t> sort_idx(boundary_facets.size());
    std::iota(sort_idx.begin(), sort_idx.end(), 0);

    std::sort(sort_idx.begin(), sort_idx.end(), 
    [&boundary_facets] (std::int32_t a, std::int32_t b)
    {
        return boundary_facets[a] < boundary_facets[b];
    });

    std::transform(sort_idx.begin(), sort_idx.end(), boundary_facets_sorted.begin(),
    [&boundary_facets] (std::int32_t idx) 
    {
        return boundary_facets[idx];
    });

    std::transform(sort_idx.begin(), sort_idx.end(), tags_sorted.begin(),
    [&tags] (std::int32_t idx) 
    {
        return tags[idx];
    });

    // Create mesh tags
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        mesh->topology(), tdim-1, boundary_facets_sorted, tags_sorted);
  
    // Model
    auto model = LinearSpectral3D<T, degreeOfBasis>(
        mesh, mt_facet, c0, rho0, sourceFrequency, sourceAmplitude,
        speedOfSound);

    // Solve
    common::Timer tsolve("Solve time");

    model.init();

    tsolve.start();
    model.rk4(startTime, finalTime, timeStepSize);
    tsolve.stop();

    if (mpi_rank == 0) {
      std::cout << "Solve time: " << tsolve.elapsed()[0] << std::endl;
      std::cout << "Time per step: " << tsolve.elapsed()[0] / numberOfStep << std::endl;
    }

    // Final solution
    auto u_n = model.u_sol();

    // Output to VTX
    dolfinx::io::VTXWriter<T> u_out(mesh->comm(), "output_final.bp", {u_n}, "BP4");
    u_out.write(0.0);

    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::average);
    
  }
}