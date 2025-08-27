import { describe, it, expect } from 'vitest'
import { latticeParamsToMatrix, fracToCartesian } from './crystal'
import { getCovalentRadius } from './utils/bonds'

describe('Crystal utilities', () => {
  it('should convert lattice parameters to matrix', () => {
    // Test with a cubic system
    const matrix = latticeParamsToMatrix(5, 5, 5, 90, 90, 90)
    expect(matrix).toEqual([
      [5, 0, 0],
      [0, 5, 0],
      [0, 0, 5]
    ])
  })

  it('should convert fractional to cartesian coordinates', () => {
    const lattice = [
      [5, 0, 0],
      [0, 5, 0],
      [0, 0, 5]
    ]
    
    const frac = [0.5, 0.5, 0.5]
    const cartesian = fracToCartesian(frac, lattice)
    
    expect(cartesian).toEqual([2.5, 2.5, 2.5])
  })
})

describe('Bonding utilities', () => {
  it('should return covalent radius for elements', () => {
    expect(getCovalentRadius('H')).toBe(0.32)
    expect(getCovalentRadius('C')).toBe(0.75)
    expect(getCovalentRadius('O')).toBe(0.66)
  })
  
  it('should return default radius for unknown elements', () => {
    expect(getCovalentRadius('Unknown')).toBe(1.0)
  })
})