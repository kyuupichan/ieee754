//===-- llvm/Support/APInt.h - For Arbitrary Precision Integer -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Sheng Zhou and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integral
// constant values and operations on them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APINT_H
#define LLVM_APINT_H

#define COMPILE_TIME_ASSERT(cond) extern int CTAssert[(cond) ? 1 : -1]

namespace llvm {

  /* An unsigned host type used as a single part of a multi-part
     bignum.  */
  __extension__ typedef unsigned long long integerPart;

  const unsigned int host_char_bit = 8;
  const unsigned int integerPartWidth = host_char_bit * sizeof(integerPart);

//===----------------------------------------------------------------------===//
//                              APInt Class
//===----------------------------------------------------------------------===//

/// @brief Class for arbitrary precision integers.
class APInt {
public:
  /// @}
  /// @name Building-block Operations for APInt and APFloat
  /// @{

  // These building block operations operate on a representation of
  // arbitrary precision, two's-complement, bignum integer values.
  // They should be sufficient to implement APInt and APFloat bignum
  // requirements.  Inputs are generally a pointer to the base of an
  // array of integer parts, representing an unsigned bignum, and a
  // count of how many parts there are.

  /// Sets the least significant part of a bignum to the input value,
  /// and zeroes out higher parts.  */
  static void tcSet(integerPart *, integerPart, unsigned int);

  /// Assign one bignum to another.
  static void tcAssign(integerPart *, const integerPart *, unsigned int);

  /// Returns true if a bignum is zero, false otherwise.
  static bool tcIsZero(const integerPart *, unsigned int);

  /// Extract the given bit of a bignum; returns 0 or 1.  Zero-based.
  static int tcExtractBit(const integerPart *, unsigned int bit);

  /// Copy the bit vector of width srcBITS from SRC, starting at bit
  /// srcLSB, to DST, of dstCOUNT parts, such that the bit srcLSB
  /// becomes the least significant bit of DST.  All high bits above
  /// srcBITS in DST are zero-filled.
  static void tcExtract(integerPart *, unsigned int dstCount, const integerPart *,
                        unsigned int srcBits, unsigned int srcLSB);

  /// Set the given bit of a bignum.  Zero-based.
  static void tcSetBit(integerPart *, unsigned int bit);

  /// Returns the bit number of the least or most significant set bit
  /// of a number.  If the input number has no bits set -1U is
  /// returned.
  static unsigned int tcLSB(const integerPart *, unsigned int);
  static unsigned int tcMSB(const integerPart *, unsigned int);

  /// Negate a bignum in-place.
  static void tcNegate(integerPart *, unsigned int);

  /// DST += RHS + CARRY where CARRY is zero or one.  Returns the
  /// carry flag.
  static integerPart tcAdd(integerPart *, const integerPart *,
			   integerPart carry, unsigned);

  /// DST -= RHS + CARRY where CARRY is zero or one.  Returns the
  /// carry flag.
  static integerPart tcSubtract(integerPart *, const integerPart *,
				integerPart carry, unsigned);

  ///  DST += SRC * MULTIPLIER + PART   if add is true
  ///  DST  = SRC * MULTIPLIER + PART   if add is false
  ///
  ///  Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
  ///  they must start at the same point, i.e. DST == SRC.
  ///
  ///  If DSTPARTS == SRC_PARTS + 1 no overflow occurs and zero is
  ///  returned.  Otherwise DST is filled with the least significant
  ///  DSTPARTS parts of the result, and if all of the omitted higher
  ///  parts were zero return zero, otherwise overflow occurred and
  ///  return one.
  static int tcMultiplyPart(integerPart *dst, const integerPart *src,
			    integerPart multiplier, integerPart carry,
			    unsigned int srcParts, unsigned int dstParts,
			    bool add);

  /// DST = LHS * RHS, where DST has the same width as the operands
  /// and is filled with the least significant parts of the result.
  /// Returns one if overflow occurred, otherwise zero.  DST must be
  /// disjoint from both operands.
  static int tcMultiply(integerPart *, const integerPart *,
			const integerPart *, unsigned);

  /// DST = LHS * RHS, where DST has width the sum of the widths of
  /// the operands.  No overflow occurs.  DST must be disjoint from
  /// both operands. Returns the number of parts required to hold the
  /// result.
  static unsigned int tcFullMultiply(integerPart *, const integerPart *,
				     const integerPart *, unsigned, unsigned);

  /// If RHS is zero LHS and REMAINDER are left unchanged, return one.
  /// Otherwise set LHS to LHS / RHS with the fractional part
  /// discarded, set REMAINDER to the remainder, return zero.  i.e.
  ///
  ///  OLD_LHS = RHS * LHS + REMAINDER
  ///
  ///  SCRATCH is a bignum of the same size as the operands and result
  ///  for use by the routine; its contents need not be initialized
  ///  and are destroyed.  LHS, REMAINDER and SCRATCH must be
  ///  distinct.
  static int tcDivide(integerPart *lhs, const integerPart *rhs,
		      integerPart *remainder, integerPart *scratch,
		      unsigned int parts);

  /// Shift a bignum left COUNT bits.  Shifted in bits are zero.
  /// There are no restrictions on COUNT.
  static void tcShiftLeft(integerPart *, unsigned int parts,
			  unsigned int count);

  /// Shift a bignum right COUNT bits.  Shifted in bits are zero.
  /// There are no restrictions on COUNT.
  static void tcShiftRight(integerPart *, unsigned int parts,
			   unsigned int count);

  /// The obvious AND, OR and XOR and complement operations.
  static void tcAnd(integerPart *, const integerPart *, unsigned int);
  static void tcOr(integerPart *, const integerPart *, unsigned int);
  static void tcXor(integerPart *, const integerPart *, unsigned int);
  static void tcComplement(integerPart *, unsigned int);
  
  /// Comparison (unsigned) of two bignums.
  static int tcCompare(const integerPart *, const integerPart *,
		       unsigned int);

  /// Increment a bignum in-place.  Return the carry flag.
  static integerPart tcIncrement(integerPart *, unsigned int);

  /// Set the least significant BITS and clear the rest.
  static void tcSetLeastSignificantBits(integerPart *, unsigned int,
					unsigned int bits);

  /// @}
};

} // End of llvm namespace

#endif
