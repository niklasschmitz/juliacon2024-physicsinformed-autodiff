using DFTK
using AtomsIO
using LinearAlgebra
using ForwardDiff
using ComponentArrays

DFTK.setup_threading()

system = load_system("./Li-BCC.xsf")
system = attach_psp(system, Dict(:Li => "hgh/lda/Li-q3"))
parsed = DFTK.parse_system(system)
cell = (; parsed.lattice, parsed.atoms, parsed.positions)

# Numerical parameters
temperature = 0.00225  # Temperature for Fermi-Dirac Smearing.
kgrid = [2, 2, 2]      # Brillouin-zone discretization

# TODO increase to 150
Ecut = 5       # Plane-wave discretization (energy cutoff)
tol = 1e-11    # SCF tolerance for density convergence

# Create supercell
supercell_size = (2, 1, 1)
supercell = DFTK.create_supercell(
    cell.lattice,
    cell.atoms,
    cell.positions,
    supercell_size
)

# # Displace second atom to get nonzero forces
# supercell = (;
#     supercell.lattice,
#     supercell.atoms,
#     positions = supercell.positions #+ [[0.01, 0, 0], [0., 0, 0]]
# )


function compute_force(ε::T; Ecut=5, tol=1e-11, symmetries=false) where {T}
    (; lattice, atoms, positions) = supercell
    pos = positions + ε * [[1., 0, 0], [0., 0, 0]]
    model = model_LDA(Matrix{T}(lattice), atoms, pos; temperature, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    response = ResponseOptions(; verbose=true)
    is_converged = DFTK.ScfConvergenceDensity(tol)
    scfres = self_consistent_field(basis; is_converged, response)
    compute_forces_cart(scfres)
end

F = compute_force(0.0)

derivative_ε_fd = let ε = 1e-5
    (compute_force(ε) - F) / ε
end
# TODO also add central difference
derivative_ε_fd_central = let ε = 1e-5
    (compute_force(ε) - compute_force(-ε)) / 2ε
end


# TODO can compare with FiniteDifferences.jl and "factor"


derivative_ε = ForwardDiff.derivative(compute_force, 0.0)

derivative_ε_sym = let ε = 1e-5
    (compute_force(ε; symmetries=true) - F) / ε
end
derivative_ε_sym = ForwardDiff.derivative(
    ε -> compute_force(ε; symmetries=true), 
    0.0
)

