/**
 * Demo structures for testing the visualizer
 */

import { CrystalStructure } from './crystal'

/**
 * Create a simple Si crystal (diamond structure) for demo
 */
export function createSiCrystal(): CrystalStructure {
  const latticeConstant = 5.43 // Å for Si
  
  // Diamond structure lattice parameters
  const lattice = {
    matrix: [
      [latticeConstant, 0, 0],
      [0, latticeConstant, 0],
      [0, 0, latticeConstant]
    ],
    a: latticeConstant,
    b: latticeConstant,
    c: latticeConstant,
    alpha: 90,
    beta: 90,
    gamma: 90
  }
  
  // Si atoms in fractional coordinates (diamond structure)
  const sites = [
    // First unit cell
    {
      element: 'Si',
      frac: [0.0, 0.0, 0.0],
      cartesian: [0.0, 0.0, 0.0],
      occupancy: 1.0,
      label: 'Si1'
    },
    {
      element: 'Si',
      frac: [0.25, 0.25, 0.25],
      cartesian: [
        0.25 * latticeConstant,
        0.25 * latticeConstant,
        0.25 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Si2'
    },
    // Second unit cell (duplicate to make visualization more interesting)
    {
      element: 'Si',
      frac: [1.0, 0.0, 0.0],
      cartesian: [latticeConstant, 0.0, 0.0],
      occupancy: 1.0,
      label: 'Si3'
    },
    {
      element: 'Si',
      frac: [1.25, 0.25, 0.25],
      cartesian: [
        1.25 * latticeConstant,
        0.25 * latticeConstant,
        0.25 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Si4'
    }
  ]
  
  return {
    lattice,
    pbc: [true, true, true],
    sites
  }
}

/**
 * Create a simple NaCl crystal for demo
 */
export function createNaClCrystal(): CrystalStructure {
  const latticeConstant = 5.64 // Å for NaCl
  
  // Cubic lattice parameters
  const lattice = {
    matrix: [
      [latticeConstant, 0, 0],
      [0, latticeConstant, 0],
      [0, 0, latticeConstant]
    ],
    a: latticeConstant,
    b: latticeConstant,
    c: latticeConstant,
    alpha: 90,
    beta: 90,
    gamma: 90
  }
  
  // NaCl atoms in fractional coordinates
  const sites = [
    // Na atoms
    {
      element: 'Na',
      frac: [0.0, 0.0, 0.0],
      cartesian: [0.0, 0.0, 0.0],
      occupancy: 1.0,
      label: 'Na1'
    },
    {
      element: 'Na',
      frac: [0.5, 0.5, 0.0],
      cartesian: [
        0.5 * latticeConstant,
        0.5 * latticeConstant,
        0.0
      ],
      occupancy: 1.0,
      label: 'Na2'
    },
    {
      element: 'Na',
      frac: [0.5, 0.0, 0.5],
      cartesian: [
        0.5 * latticeConstant,
        0.0,
        0.5 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Na3'
    },
    {
      element: 'Na',
      frac: [0.0, 0.5, 0.5],
      cartesian: [
        0.0,
        0.5 * latticeConstant,
        0.5 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Na4'
    },
    // Cl atoms
    {
      element: 'Cl',
      frac: [0.5, 0.5, 0.5],
      cartesian: [
        0.5 * latticeConstant,
        0.5 * latticeConstant,
        0.5 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Cl1'
    },
    {
      element: 'Cl',
      frac: [0.0, 0.0, 0.5],
      cartesian: [
        0.0,
        0.0,
        0.5 * latticeConstant
      ],
      occupancy: 1.0,
      label: 'Cl2'
    },
    {
      element: 'Cl',
      frac: [0.0, 0.5, 0.0],
      cartesian: [
        0.0,
        0.5 * latticeConstant,
        0.0
      ],
      occupancy: 1.0,
      label: 'Cl3'
    },
    {
      element: 'Cl',
      frac: [0.5, 0.0, 0.0],
      cartesian: [
        0.5 * latticeConstant,
        0.0,
        0.0
      ],
      occupancy: 1.0,
      label: 'Cl4'
    }
  ]
  
  return {
    lattice,
    pbc: [true, true, true],
    sites
  }
}