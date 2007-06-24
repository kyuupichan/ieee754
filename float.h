/*
   Copyright 2004-2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#ifndef LLVM_FLOAT_H
#define LLVM_FLOAT_H

#define HOST_CHAR_BIT 8
#define compileTimeAssert(cond) extern int CTAssert[(cond) ? 1 : -1]
#define integerPartWidth (HOST_CHAR_BIT * sizeof (llvm::integerPart))

namespace llvm {

  /* The most convenient unsigned host type.  */
   __extension__ typedef unsigned long long integerPart;

  /* Exponents are stored as signed numbers.  */
  typedef signed short exponent_t;

  struct fltSemantics;
  struct decimal_number;

  enum e_lost_fraction {
    lf_exactly_zero,
    lf_less_than_half,
    lf_exactly_half,
    lf_more_than_half
  };

class APInt {
 public:
  /* Sets the least significant part of a bignum to the input value,
     and zeroes out higher parts.  */
  static void tcSet (integerPart *, integerPart, unsigned int);

  /* Assign one bignum to another.  */
  static void tcAssign (integerPart *, const integerPart *, unsigned int);

  /* Returns true if a bignum is zero, false otherwise.  */
  static bool tcIsZero (const integerPart *, unsigned int);

  /* Extract the given bit of a bignum; returns 0 or 1.  BIT cannot be
     zero.  */
  static int tcExtractBit (const integerPart *, unsigned int bit);

  /* Set the given bit of a bignum.  BIT cannot be zero.  */
  static void tcSetBit (integerPart *, unsigned int bit);

  /* Returns the bit  number of the least or  most significant set bit
     of  a number.   If  the input  number  has no  bits  set zero  is
     returned.  */
  static unsigned int tcLSB (const integerPart *, unsigned int);
  static unsigned int tcMSB (const integerPart *, unsigned int);

  /* Negate a bignum in-place.  */
  static void tcNegate (integerPart *, unsigned int);

  /* DST += RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static integerPart tcAdd (integerPart *, const integerPart *,
			    integerPart carry, unsigned);

  /* DST -= RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static integerPart tcSubtract (integerPart *, const integerPart *,
				 integerPart carry, unsigned);

  /*  DST += SRC * MULTIPLIER + PART   if add is true
      DST  = SRC * MULTIPLIER + PART   if add is false

      Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
      they must start at the same point, i.e. DST == SRC.

      If DSTPARTS == SRC_PARTS + 1 no overflow occurs and zero is
      returned.  Otherwise DST is filled with the least significant
      DSTPARTS parts of the result, and if all of the omitted higher
      parts were zero return zero, otherwise overflow occurred and
      return one.  */
  static int tcMultiplyPart (integerPart *dst, const integerPart *src,
			     integerPart multiplier, integerPart carry,
			     unsigned int srcParts, unsigned int dstParts,
			     bool add);

  /* DST = LHS * RHS, where DST has the same width as the operands and
     is filled with the least significant parts of the result.
     Returns one if overflow occurred, otherwise zero.  DST must be
     disjoint from both operands.  */
  static int tcMultiply (integerPart *, const integerPart *,
			 const integerPart *, unsigned);

  /* DST = LHS * RHS, where DST has twice the width as the operands.  No
     overflow occurs.  DST must be disjoint from both operands.  */
  static void tcFullMultiply (integerPart *, const integerPart *,
			      const integerPart *, unsigned);

  /* If RHS is zero LHS and REMAINDER are left unchanged, return one.
     Otherwise set LHS to LHS / RHS with the fractional part
     discarded, set REMAINDER to the remainder, return zero.  i.e.

       OLD_LHS = RHS * LHS + REMAINDER

     SCRATCH is a bignum of the same size as the operands and result
     for use by the routine; its contents need not be initialized and
     are destroyed.  LHS, REMAINDER and SCRATCH must be distinct.  */
  static int tcDivide (integerPart *lhs, const integerPart *rhs,
		       integerPart *remainder, integerPart *scratch,
		       unsigned int parts);

  /* Shift a bignum left COUNT bits.  Shifted in bits are zero.  There
     are no restrictions on COUNT.  */
  static void tcShiftLeft (integerPart *, unsigned int parts,
			   unsigned int count);

  /* Shift a bignum right COUNT bits.  Shifted in bits are zero.
     There are no restrictions on COUNT.  */
  static void tcShiftRight (integerPart *, unsigned int parts,
			    unsigned int count);

  /* The obvious AND, OR and XOR and complement operations.  */
  static void tcAnd (integerPart *, const integerPart *, unsigned int);
  static void tcOr (integerPart *, const integerPart *, unsigned int);
  static void tcXor (integerPart *, const integerPart *, unsigned int);
  static void tcComplement (integerPart *, unsigned int);
  
  /* Comparison (unsigned) of two bignums.  */
  static int tcCompare (const integerPart *, const integerPart *,
			unsigned int);

  /* Increment a bignum in-place.  Return the carry flag.  */
  static integerPart tcIncrement (integerPart *, unsigned int);

  /* Set the least significant BITS and clear the rest.  */
  static void tcSetLeastSignificantBits (integerPart *, unsigned int,
					 unsigned int bits);
};

class APFloat {
 public:

  /* We support the following floating point semantics.  */
  static const fltSemantics ieee_single;
  static const fltSemantics ieee_double;
  static const fltSemantics ieee_quad;
  static const fltSemantics x87_double_extended;

  static unsigned int semanticsPrecision (const fltSemantics &);

  /* Floating point numbers have a four-state comparison relation.  */
  enum cmpResult {
    cmpLessThan,
    cmpEqual,
    cmpGreaterThan,
    cmpUnordered
  };

  /* IEEE-754R gives five rounding modes.  */
  enum roundingMode {
    rmNearestTiesToEven,
    rmTowardPositive,
    rmTowardNegative,
    rmTowardZero,
    rmNearestTiesToAway
  };

  /* Operation status.  opUnderflow or opOverflow are always returned
     or-ed with opInexact.  */
  enum opStatus {
    opOK          = 0x00,
    opInvalidOp   = 0x01,
    opDivByZero   = 0x02,
    opOverflow    = 0x04,
    opUnderflow   = 0x08,
    opInexact     = 0x10
  };

  /* Category of internally-represented number.  */
  enum fltCategory {
    fcInfinity,
    fcNaN,
    fcNormal,
    fcZero
  };

  /* Constructors.  */
  APFloat (const fltSemantics &, const char *);
  APFloat (const fltSemantics &, integerPart);
  APFloat (const fltSemantics &, fltCategory, bool negative);
  APFloat (const APFloat &);
  ~APFloat ();

  /* Arithmetic.  */
  opStatus add (const APFloat &, roundingMode);
  opStatus subtract (const APFloat &, roundingMode);
  opStatus multiply (const APFloat &, roundingMode);
  opStatus divide (const APFloat &, roundingMode);
  opStatus fusedMultiplyAdd (const APFloat &, const APFloat &, roundingMode);
  void changeSign ();

  /* Conversions.  */
  opStatus convert (const fltSemantics &, roundingMode);
  opStatus convertToInteger (integerPart *, unsigned int, bool,
			     roundingMode) const;
  opStatus convertFromInteger (const integerPart *, unsigned int, bool,
			       roundingMode);
  opStatus convertFromString (const char *, roundingMode);

  /* Comparison with another floating point number.  */
  cmpResult compare (const APFloat &) const;

  /* Simple queries.  */
  fltCategory getCategory () const { return category; }
  const fltSemantics &getSemantics () const { return *semantics; }
  bool isZero () const { return category == fcZero; }
  bool isNonZero () const { return category != fcZero; }
  bool isNegative () const { return sign; }

  APFloat& operator= (const APFloat &);

 private:

  /* Trivial queries.  */
  integerPart *significandParts ();
  const integerPart *significandParts () const;
  unsigned int partCount () const;

  /* Significand operations.  */
  integerPart addSignificand (const APFloat &);
  integerPart subtractSignificand (const APFloat &, integerPart);
  e_lost_fraction addOrSubtractSignificand (const APFloat &, bool subtract);
  e_lost_fraction multiplySignificand (const APFloat &, const APFloat *);
  e_lost_fraction divideSignificand (const APFloat &);
  void incrementSignificand ();
  void initialize (const fltSemantics *);
  void shiftSignificandLeft (unsigned int);
  e_lost_fraction shiftSignificandRight (unsigned int);
  unsigned int significandLSB () const;
  unsigned int significandMSB () const;
  void zeroSignificand ();

  /* Arithmetic on special values.  */
  opStatus addOrSubtractSpecials (const APFloat &, bool subtract);
  opStatus divideSpecials (const APFloat &);
  opStatus multiplySpecials (const APFloat &);

  /* Miscellany.  */
  opStatus normalize (roundingMode, e_lost_fraction);
  opStatus addOrSubtract (const APFloat &, roundingMode, bool subtract);
  cmpResult compareAbsoluteValue (const APFloat &) const;
  opStatus handleOverflow (roundingMode);
  bool roundAwayFromZero (roundingMode, e_lost_fraction);
  opStatus convertFromUnsignedInteger (integerPart *, unsigned int,
				       roundingMode);
  e_lost_fraction combineLostFractions (e_lost_fraction, e_lost_fraction);
  opStatus convertFromHexadecimalString (const char *, roundingMode);

  void assign (const APFloat &);
  void copySignificand (const APFloat &);
  void freeSignificand ();

  /* What kind of semantics does this value obey?  */
  const fltSemantics *semantics;

  /* Significand - the fraction with an explicit integer bit.  Must be
     at least one bit wider than the target precision.  */
  union Significand
  {
    integerPart part;
    integerPart *parts;
  } significand;

  /* The exponent - a signed number.  */
  exponent_t exponent;

  /* What kind of floating point number this is.  */
  fltCategory category: 2;

  /* The sign bit of this number.  */
  unsigned int sign: 1;
};

}

#endif /* LLVM_FLOAT_H */
