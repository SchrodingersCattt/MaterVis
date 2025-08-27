/**
 * CIF (Crystallographic Information File) parser
 */

import { CrystalStructure, latticeParamsToMatrix, fracToCartesian } from '../crystal'

export class CIFParser {
  /**
   * Parse CIF text content into a CrystalStructure object
   * @param cifText The CIF file content as string
   * @returns Parsed CrystalStructure
   */
  public static parse(cifText: string): CrystalStructure {
    // This is a simplified parser - in a real implementation, 
    // we would need a more robust CIF parser
    
    const lines = cifText.split('\n')
    let latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number } | null = null
    const sites: any[] = []
    let loopStarted = false
    let atomFields: string[] = []
    
    // Parse the CIF line by line
    for (const line of lines) {
      const trimmed = line.trim()
      
      // Skip empty lines and comments
      if (!trimmed || trimmed.startsWith('#')) continue
      
      // Parse lattice parameters
      if (trimmed.startsWith('_cell_length_a')) {
        const a = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a, b: 0, c: 0, alpha: 0, beta: 0, gamma: 0 }
        else latticeParams.a = a
      } 
      else if (trimmed.startsWith('_cell_length_b')) {
        const b = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a: 0, b, c: 0, alpha: 0, beta: 0, gamma: 0 }
        else latticeParams.b = b
      }
      else if (trimmed.startsWith('_cell_length_c')) {
        const c = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a: 0, b: 0, c, alpha: 0, beta: 0, gamma: 0 }
        else latticeParams.c = c
      }
      else if (trimmed.startsWith('_cell_angle_alpha')) {
        const alpha = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a: 0, b: 0, c: 0, alpha, beta: 0, gamma: 0 }
        else latticeParams.alpha = alpha
      }
      else if (trimmed.startsWith('_cell_angle_beta')) {
        const beta = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a: 0, b: 0, c: 0, alpha: 0, beta, gamma: 0 }
        else latticeParams.beta = beta
      }
      else if (trimmed.startsWith('_cell_angle_gamma')) {
        const gamma = parseFloat(trimmed.split(/\s+/)[1])
        if (!latticeParams) latticeParams = { a: 0, b: 0, c: 0, alpha: 0, beta: 0, gamma }
        else latticeParams.gamma = gamma
      }
      
      // Parse atom positions
      if (trimmed.startsWith('loop_')) {
        loopStarted = true
        atomFields = []
        continue
      }
      
      if (loopStarted && trimmed.startsWith('_')) {
        atomFields.push(trimmed)
        continue
      }
      
      if (loopStarted && atomFields.length > 0 && !trimmed.startsWith('_')) {
        // This is atom data
        const values = trimmed.split(/\s+/)
        if (values.length >= 5) {
          sites.push({
            element: values[0],
            frac: [
              parseFloat(values[values.length - 3]),
              parseFloat(values[values.length - 2]),
              parseFloat(values[values.length - 1])
            ],
            occupancy: values.length > 5 ? parseFloat(values[values.length - 4]) : 1.0
          })
        }
      }
      
      // Reset when we hit another loop or non-data line
      if (trimmed.startsWith('_') && !trimmed.startsWith('_atom')) {
        loopStarted = false
      }
    }
    
    // Create the lattice matrix
    if (!latticeParams) {
      throw new Error('Missing lattice parameters in CIF file')
    }
    
    const latticeMatrix = latticeParamsToMatrix(
      latticeParams.a,
      latticeParams.b,
      latticeParams.c,
      latticeParams.alpha,
      latticeParams.beta,
      latticeParams.gamma
    )
    
    // Convert fractional to cartesian coordinates
    sites.forEach(site => {
      site.cartesian = fracToCartesian(site.frac, latticeMatrix)
    })
    
    const crystal: CrystalStructure = {
      lattice: {
        matrix: latticeMatrix,
        a: latticeParams.a,
        b: latticeParams.b,
        c: latticeParams.c,
        alpha: latticeParams.alpha,
        beta: latticeParams.beta,
        gamma: latticeParams.gamma
      },
      pbc: [true, true, true],
      sites: sites
    }
    
    return crystal
  }
}