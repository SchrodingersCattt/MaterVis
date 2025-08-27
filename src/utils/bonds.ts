/**
 * Bonding calculation utilities
 */

import { CrystalStructure } from '../crystal'
import { periodicDistance } from './lattice'

// Covalent radii (in Å) - Pyykkö & Atsumi values
const COVALENT_RADII: { [element: string]: number } = {
  'H': 0.32,
  'He': 0.46,
  'Li': 1.33,
  'Be': 1.02,
  'B': 0.85,
  'C': 0.75,
  'N': 0.71,
  'O': 0.66,
  'F': 0.57,
  'Ne': 0.58,
  'Na': 1.55,
  'Mg': 1.39,
  'Al': 1.26,
  'Si': 1.16,
  'P': 1.11,
  'S': 1.03,
  'Cl': 0.99,
  'Ar': 1.07,
  'K': 1.96,
  'Ca': 1.71,
  'Sc': 1.48,
  'Ti': 1.36,
  'V': 1.34,
  'Cr': 1.22,
  'Mn': 1.19,
  'Fe': 1.16,
  'Co': 1.11,
  'Ni': 1.10,
  'Cu': 1.12,
  'Zn': 1.18,
  'Ga': 1.24,
  'Ge': 1.21,
  'As': 1.21,
  'Se': 1.16,
  'Br': 1.14,
  'Kr': 1.17,
  // Add more elements as needed
}

export interface Bond {
  atom1: number // index of first atom
  atom2: number // index of second atom
  distance: number
  cutoff: number
}

/**
 * Calculate bonding based on covalent radii
 * @param crystal Crystal structure
 * @param scaleFactor Scaling factor for covalent radii sum
 * @param minCut Minimum bond length
 * @param maxCut Maximum bond length
 * @param pairOverrides Custom pair cutoffs
 * @returns Array of bonds
 */
export function calculateBonds(
  crystal: CrystalStructure,
  scaleFactor: number = 1.1,
  minCut: number = 0.7,
  maxCut: number = 3.2,
  pairOverrides?: { [key: string]: number }
): Bond[] {
  const bonds: Bond[] = []
  const latticeMatrix = crystal.lattice.matrix
  
  // For simplicity, we'll only check bonds within the unit cell and nearest neighbors
  // In a full implementation, we would need to consider all periodic images within a cutoff
  
  for (let i = 0; i < crystal.sites.length; i++) {
    const site1 = crystal.sites[i]
    const element1 = site1.element
    
    for (let j = i + 1; j < crystal.sites.length; j++) {
      const site2 = crystal.sites[j]
      const element2 = site2.element
      
      // Get cutoff distance
      let cutoff = minCut
      
      // Check for pair override
      const pairKey1 = `${element1}-${element2}`
      const pairKey2 = `${element2}-${element1}`
      
      if (pairOverrides && (pairOverrides[pairKey1] || pairOverrides[pairKey2])) {
        cutoff = pairOverrides[pairKey1] || pairOverrides[pairKey2] || cutoff
      } else {
        // Calculate cutoff based on covalent radii
        const r1 = COVALENT_RADII[element1] || 1.0
        const r2 = COVALENT_RADII[element2] || 1.0
        cutoff = scaleFactor * (r1 + r2)
        
        // Apply min/max limits
        cutoff = Math.max(minCut, Math.min(maxCut, cutoff))
      }
      
      // Calculate distance with periodic boundary conditions
      const distance = periodicDistance(site1.frac, site2.frac, latticeMatrix)
      
      // If within cutoff, add bond
      if (distance <= cutoff) {
        bonds.push({
          atom1: i,
          atom2: j,
          distance: distance,
          cutoff: cutoff
        })
      }
    }
  }
  
  return bonds
}

/**
 * Get covalent radius for an element
 * @param element Element symbol
 * @returns Covalent radius in Å
 */
export function getCovalentRadius(element: string): number {
  return COVALENT_RADII[element] || 1.0
}