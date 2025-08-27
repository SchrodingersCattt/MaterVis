/**
 * Crystallographic math utilities
 */

/**
 * Apply periodic boundary conditions to fractional coordinates
 * @param frac Fractional coordinates [x, y, z]
 * @returns Wrapped fractional coordinates in [0, 1) range
 */
export function wrapFractional(frac: number[]): number[] {
  return frac.map(f => f - Math.floor(f))
}

/**
 * Apply minimum image convention to fractional coordinates
 * @param frac Fractional distance vector
 * @returns Wrapped fractional distance vector in [-0.5, 0.5) range
 */
export function minimumImage(frac: number[]): number[] {
  return frac.map(f => f - Math.round(f))
}

/**
 * Calculate distance between two points with periodic boundary conditions
 * @param frac1 First point in fractional coordinates
 * @param frac2 Second point in fractional coordinates
 * @param latticeMatrix Lattice matrix
 * @returns Distance in Å
 */
export function periodicDistance(frac1: number[], frac2: number[], latticeMatrix: number[][]): number {
  const df = [
    frac2[0] - frac1[0],
    frac2[1] - frac1[1],
    frac2[2] - frac1[2]
  ]
  
  // Apply minimum image convention
  const dfWrapped = minimumImage(df)
  
  // Convert to cartesian
  const dr = [
    dfWrapped[0] * latticeMatrix[0][0] + dfWrapped[1] * latticeMatrix[1][0] + dfWrapped[2] * latticeMatrix[2][0],
    dfWrapped[0] * latticeMatrix[0][1] + dfWrapped[1] * latticeMatrix[1][1] + dfWrapped[2] * latticeMatrix[2][1],
    dfWrapped[0] * latticeMatrix[0][2] + dfWrapped[1] * latticeMatrix[1][2] + dfWrapped[2] * latticeMatrix[2][2]
  ]
  
  // Calculate distance
  return Math.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
}

/**
 * Generate supercell coordinates
 * @param size Supercell size [nx, ny, nz]
 * @returns Array of translation vectors for supercell
 */
export function generateSupercellTranslations(size: [number, number, number]): number[][] {
  const translations: number[][] = []
  
  for (let x = 0; x < size[0]; x++) {
    for (let y = 0; y < size[1]; y++) {
      for (let z = 0; z < size[2]; z++) {
        translations.push([x, y, z])
      }
    }
  }
  
  return translations
}