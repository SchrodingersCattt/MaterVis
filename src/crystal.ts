/**
 * Represents a crystal structure with lattice, atoms, and associated properties
 */

export interface Lattice {
  matrix: number[][]
  a: number
  b: number
  c: number
  alpha: number
  beta: number
  gamma: number
}

export interface Site {
  element: string
  frac: number[] // fractional coordinates [x, y, z]
  occupancy: number
  label?: string
  disorder_group?: string
  oxidation_state?: number
  cartesian?: number[] // cartesian coordinates [x, y, z]
}

export interface SiteProperties {
  [key: string]: any[]
}

export interface BondingSettings {
  mode: 'covalent' | 'ionic' | 'vdw' | 'custom'
  f: number // scaling factor
  min: number // minimum bond length
  max: number // maximum bond length
  pairOverrides?: { [key: string]: number } // custom pair distances
}

export interface StyleSettings {
  colors: { [element: string]: string }
  bonding: BondingSettings
}

export interface CrystalStructure {
  lattice: Lattice
  pbc: boolean[] // periodic boundary conditions [a, b, c]
  sites: Site[]
  site_properties?: SiteProperties
  style?: StyleSettings
}

/**
 * Convert lattice parameters to matrix representation
 */
export function latticeParamsToMatrix(a: number, b: number, c: number, alpha: number, beta: number, gamma: number): number[][] {
  // Convert angles from degrees to radians
  const alphaRad = alpha * Math.PI / 180
  const betaRad = beta * Math.PI / 180
  const gammaRad = gamma * Math.PI / 180
  
  // Calculate matrix components
  const bx = b * Math.cos(gammaRad)
  const by = b * Math.sin(gammaRad)
  
  const cx = c * Math.cos(betaRad)
  const cy = c * (Math.cos(alphaRad) - Math.cos(betaRad) * Math.cos(gammaRad)) / Math.sin(gammaRad)
  const cz = Math.sqrt(c * c - cx * cx - cy * cy)
  
  return [
    [a, 0, 0],
    [bx, by, 0],
    [cx, cy, cz]
  ]
}

/**
 * Convert fractional coordinates to cartesian
 */
export function fracToCartesian(frac: number[], latticeMatrix: number[][]): number[] {
  return [
    frac[0] * latticeMatrix[0][0] + frac[1] * latticeMatrix[1][0] + frac[2] * latticeMatrix[2][0],
    frac[0] * latticeMatrix[0][1] + frac[1] * latticeMatrix[1][1] + frac[2] * latticeMatrix[2][1],
    frac[0] * latticeMatrix[0][2] + frac[1] * latticeMatrix[1][2] + frac[2] * latticeMatrix[2][2]
  ]
}