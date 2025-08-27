/**
 * POSCAR/CONTCAR (VASP format) parser
 */

import { CrystalStructure, latticeParamsToMatrix, fracToCartesian } from '../crystal'

export class POSCARParser {
  /**
   * Parse POSCAR text content into a CrystalStructure object
   * @param poscarText The POSCAR file content as string
   * @returns Parsed CrystalStructure
   */
  public static parse(poscarText: string): CrystalStructure {
    const lines = poscarText.split('\n').map(line => line.trim()).filter(line => line.length > 0)
    
    if (lines.length < 8) {
      throw new Error('Invalid POSCAR format: insufficient lines')
    }
    
    // Line 1: Comment (optional)
    // Line 2: Scale factor
    const scaleFactor = parseFloat(lines[1])
    
    // Lines 3-5: Lattice vectors
    const latticeVectors = [
      lines[2].split(/\s+/).map(parseFloat),
      lines[3].split(/\s+/).map(parseFloat),
      lines[4].split(/\s+/).map(parseFloat)
    ]
    
    // Scale the lattice vectors
    const scaledLattice = latticeVectors.map(vec => 
      vec.map(val => val * scaleFactor)
    )
    
    // Line 6: Elements list (if present) or number of atoms
    const elementsLine = lines[5].split(/\s+/)
    let elements: string[] = []
    let elementCounts: number[] = []
    let lineIndex = 6
    
    // Check if line 6 contains element symbols or numbers
    if (isNaN(parseInt(elementsLine[0]))) {
      // Element symbols are present
      elements = elementsLine
      elementCounts = lines[6].split(/\s+/).map(parseInt)
      lineIndex = 7
    } else {
      // Only number of atoms, use generic names
      elementCounts = elementsLine.map(val => parseInt(val))
      elements = elementCounts.map((_, i) => `Element${i+1}`)
    }
    
    // Line with "Selective dynamics" (optional)
    let selectiveDynamics = false
    if (lines[lineIndex].toLowerCase().startsWith('s')) {
      selectiveDynamics = true
      lineIndex++
    }
    
    // Coordinate system (Direct/Cartesian)
    const direct = lines[lineIndex].toLowerCase().startsWith('d')
    lineIndex++
    
    // Parse atomic positions
    const sites = []
    let atomIndex = 0
    
    for (let i = 0; i < elements.length; i++) {
      const element = elements[i]
      const count = elementCounts[i]
      
      for (let j = 0; j < count; j++) {
        if (lineIndex + atomIndex >= lines.length) {
          throw new Error('Not enough atomic positions in POSCAR')
        }
        
        const coords = lines[lineIndex + atomIndex].split(/\s+/).slice(0, 3).map(parseFloat)
        atomIndex++
        
        const site: any = {
          element: element,
          frac: direct ? coords : this.cartesianToFractional(coords, scaledLattice),
          occupancy: 1.0
        }
        
        // If in cartesian mode, also store cartesian coordinates
        if (!direct) {
          site.cartesian = coords
        } else {
          site.cartesian = fracToCartesian(site.frac, scaledLattice)
        }
        
        sites.push(site)
      }
    }
    
    // Calculate lattice parameters from matrix
    const a = Math.sqrt(scaledLattice[0][0]**2 + scaledLattice[0][1]**2 + scaledLattice[0][2]**2)
    const b = Math.sqrt(scaledLattice[1][0]**2 + scaledLattice[1][1]**2 + scaledLattice[1][2]**2)
    const c = Math.sqrt(scaledLattice[2][0]**2 + scaledLattice[2][1]**2 + scaledLattice[2][2]**2)
    
    // Calculate angles
    const alpha = this.calculateAngle(scaledLattice[1], scaledLattice[2]) // between b and c
    const beta = this.calculateAngle(scaledLattice[0], scaledLattice[2])  // between a and c
    const gamma = this.calculateAngle(scaledLattice[0], scaledLattice[1]) // between a and b
    
    const crystal: CrystalStructure = {
      lattice: {
        matrix: scaledLattice,
        a,
        b,
        c,
        alpha: alpha * 180 / Math.PI,
        beta: beta * 180 / Math.PI,
        gamma: gamma * 180 / Math.PI
      },
      pbc: [true, true, true],
      sites: sites
    }
    
    return crystal
  }
  
  /**
   * Convert cartesian coordinates to fractional
   */
  private static cartesianToFractional(cart: number[], lattice: number[][]): number[] {
    // Calculate the determinant of the lattice matrix
    const det = lattice[0][0] * (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1]) -
                lattice[0][1] * (lattice[1][0] * lattice[2][2] - lattice[1][2] * lattice[2][0]) +
                lattice[0][2] * (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0])
    
    // Calculate the inverse lattice matrix
    const invLattice = [
      [
        (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1]) / det,
        (lattice[0][2] * lattice[2][1] - lattice[0][1] * lattice[2][2]) / det,
        (lattice[0][1] * lattice[1][2] - lattice[0][2] * lattice[1][1]) / det
      ],
      [
        (lattice[1][2] * lattice[2][0] - lattice[1][0] * lattice[2][2]) / det,
        (lattice[0][0] * lattice[2][2] - lattice[0][2] * lattice[2][0]) / det,
        (lattice[0][2] * lattice[1][0] - lattice[0][0] * lattice[1][2]) / det
      ],
      [
        (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0]) / det,
        (lattice[0][1] * lattice[2][0] - lattice[0][0] * lattice[2][1]) / det,
        (lattice[0][0] * lattice[1][1] - lattice[0][1] * lattice[1][0]) / det
      ]
    ]
    
    // Multiply cartesian coordinates by inverse lattice matrix
    return [
      cart[0] * invLattice[0][0] + cart[1] * invLattice[1][0] + cart[2] * invLattice[2][0],
      cart[0] * invLattice[0][1] + cart[1] * invLattice[1][1] + cart[2] * invLattice[2][1],
      cart[0] * invLattice[0][2] + cart[1] * invLattice[1][2] + cart[2] * invLattice[2][2]
    ]
  }
  
  /**
   * Calculate angle between two vectors in radians
   */
  private static calculateAngle(vec1: number[], vec2: number[]): number {
    const dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    const mag1 = Math.sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
    const mag2 = Math.sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
    return Math.acos(dot / (mag1 * mag2))
  }
}