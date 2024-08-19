from openmm import *
from openmm.app import *
from openmm.unit import *

def set_simulation(args, forcefield_files, start_pdb, platform, properties):
    # Set force field, system, integrator for simulation
    forcefield = ForceField(*forcefield_files)
    system = forcefield.createSystem(
        start_pdb.topology,
        nonbondedCutoff=3 * nanometer,
        constraints=HBonds
    )
    integrator = LangevinIntegrator(
        args.temperature * kelvin,
        1 / picosecond,
        1 * femtoseconds
    )
    
    # Create simulation
    simulation = Simulation(
        start_pdb.topology,
        system,
        integrator,
        platform,
        properties
    )
    simulation.context.setPositions(start_pdb.positions)
    simulation.minimizeEnergy()
    
    return simulation