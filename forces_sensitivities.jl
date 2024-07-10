using DFTK
using AtomsIO
using LinearAlgebra
using JLD2
using ForwardDiff
using ComponentArrays

DFTK.setup_threading()

system = load_system("./Li-BCC.xsf")
system = attach_psp(system, Dict(:Li => "hgh/lda/Li-q3"))
parsed = DFTK.parse_system(system)
cell = (; parsed.lattice, parsed.atoms, parsed.positions)

# Numerical parameters
temperature = 0.00225  # Temperature for Fermi-Dirac Smearing.
kgrid = [2, 4, 4]      # Brillouin-zone discretization
Ecut = 150       # Plane-wave discretization (energy cutoff)
tol = 1e-8    # SCF tolerance for density convergence

# Create supercell
supercell_size = (2, 1, 1)
supercell = DFTK.create_supercell(
    cell.lattice,
    cell.atoms,
    cell.positions,
    supercell_size
)

function run_scf(ε::T; Ecut=5, tol=1e-8, symmetries=false) where {T}
    (; lattice, atoms, positions) = supercell
    pos = positions + ε * [[1., 0, 0], [0., 0, 0]]
    model = model_LDA(Matrix{T}(lattice), atoms, pos; temperature, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    response = ResponseOptions(; verbose=true)
    is_converged = DFTK.ScfConvergenceDensity(tol)
    self_consistent_field(basis; is_converged, response)
end

compute_force(ε; kwargs...) = compute_forces_cart(run_scf(ε; kwargs...))

F = compute_force(0.0)

derivative_ε_fd = let ε = 1e-5
    (compute_force(ε) - F) / ε
end
# TODO also add central difference
derivative_ε_fd_central = let ε = 1e-5
    (compute_force(ε) - compute_force(-ε)) / 2ε
end

scfres = run_scf(0.1)
save_scfres("scfres_dummy.jld2", scfres; save_ψ=true)


# TODO can compare with FiniteDifferences.jl and "factor"

derivative_ε = ForwardDiff.derivative(compute_force, 0.0)

derivative_ε_sym = let ε = 1e-5
    (compute_force(ε; symmetries=true) - F) / ε
end
derivative_ε_sym = ForwardDiff.derivative(
    ε -> compute_force(ε; symmetries=true), 
    0.0
)

function compute_force_ca(ε::T; Ecut=5, tol=1e-8, symmetries=false) where {T}
    (; lattice, atoms, positions) = supercell
    pos = positions + ε * [[1., 0, 0], [0., 0, 0]]
    model = model_LDA(Matrix{T}(lattice), atoms, pos; temperature, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    response = ResponseOptions(; verbose=true)
    is_converged = DFTK.ScfConvergenceDensity(tol)
    scfres = self_consistent_field(basis; is_converged, response)
    forces = compute_forces_cart(scfres)
    ComponentArray(
        forces=forces,
        ρ=scfres.ρ,
        energies=collect(values(scfres.energies)),
        εF=scfres.εF,
        occupation=reduce(vcat, scfres.occupation),
    )
end

using DifferentiationInterface


for Ecut in 5:5:20
    scfres = run_scf(0.0; Ecut)  # TODO duplication is not necessary here.
    F, dF = value_and_derivative(ε -> compute_force_ca(ε; Ecut), AutoForwardDiff(), 0.0)
    save_scfres("scf/Li-BCC-Ecut$Ecut-scfres.jld2", scfres;
                save_ψ=false, extra_data=Dict("F"=>F, "dF"=>dF))
end

