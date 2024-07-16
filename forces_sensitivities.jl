using DFTK
using AtomsIO
using LinearAlgebra
using JLD2
using ForwardDiff
using ComponentArrays
using DifferentiationInterface
DFTK.setup_threading()


system = load_system("./Li-BCC.xsf")
system = attach_psp(system, Dict(:Li => "hgh/lda/Li-q3"))
parsed = DFTK.parse_system(system)
cell = (; parsed.lattice, parsed.atoms, parsed.positions)

# Numerical parameters
temperature = 0.00225  # Temperature for Fermi-Dirac Smearing.
kgrid = (2, 4, 4)      # Brillouin-zone discretization
# Ecut = 150           # Plane-wave discretization (energy cutoff)
# tol = 1e-8           # SCF tolerance for density convergence

# Create supercell
supercell_size = (2, 1, 1)
supercell = DFTK.create_supercell(
    cell.lattice,
    cell.atoms,
    cell.positions,
    supercell_size
)

function run_scf(ε::T; Ecut, tol=1e-8, symmetries=false) where {T}
    (; lattice, atoms, positions) = supercell
    pos = positions + ε * [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    model = model_PBE(Matrix{T}(lattice), atoms, pos; temperature, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    response = ResponseOptions(; verbose=true)
    is_converged = DFTK.ScfConvergenceDensity(tol)
    self_consistent_field(basis; is_converged, response)
end

function compute_quantities(ε::T; Ecut, tol=1e-8, symmetries=false) where {T}
    scfres = run_scf(ε; Ecut, tol, symmetries)
    forces = compute_forces_cart(scfres)
    ComponentArray(
        forces=forces,
        ρ=scfres.ρ,
        energies=collect(values(scfres.energies)),
        εF=scfres.εF,
        occupation=reduce(vcat, scfres.occupation),
    )
end



mkpath("scf")
for Ecut in 5:5:50
    @info "" Ecut
    scfres = run_scf(0.0; Ecut)  # TODO avoid this duplication.

    # Implicit differentiation
    F, dF = value_and_derivative(ε -> compute_quantities(ε; Ecut), AutoForwardDiff(), 0.0)
    save_scfres("scf/Li-BCC-Ecut$Ecut-scfres.jld2", scfres;
        save_ψ=false, extra_data=Dict("F" => F, "dF" => dF))
end
