//===-- APFloat.cpp - Implement APFloat class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Neil Booth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include "APFloat.h"

using namespace llvm;

#define convolve(lhs, rhs) ((lhs) * 4 + (rhs))

/* Assumed in hexadecimal significand parsing, and conversion to
   hexadecimal strings.  */
COMPILE_TIME_ASSERT(integerPartWidth % 4 == 0);

namespace llvm {

  /* Represents floating point arithmetic semantics.  */
  struct fltSemantics {
    /* The largest E such that 2^E is representable; this matches the
       definition of IEEE 754.  */
    exponent_t maxExponent;

    /* The smallest E such that 2^E is a normalized number; this
       matches the definition of IEEE 754.  */
    exponent_t minExponent;

    /* Number of bits in the significand.  This includes the integer
       bit.  */
    unsigned int precision;
  };

  const fltSemantics APFloat::IEEEsingle = { 127, -126, 24 };
  const fltSemantics APFloat::IEEEdouble = { 1023, -1022, 53 };
  const fltSemantics APFloat::IEEEquad = { 16383, -16382, 113 };
  const fltSemantics APFloat::x87DoubleExtended = { 16383, -16382, 64 };

  /* A tight upper bound on number of parts required to hold the value
     pow(5, power) is

       power * 815 / (351 * integerPartWidth) + 1

     However, whilst the result may require only this many parts,
     because we are multiplying two values to get it, the
     multiplication may require an extra part with the excess part
     being zero (consider the trivial case of 1 * 1, tcFullMultiply
     requires two parts to hold the single-part result).  So we add an
     extra one to guarantee enough space whilst multiplying.  */
  const unsigned int maxExponent = 16383;
  const unsigned int maxPrecision = 113;
  const unsigned int maxPowerOfFiveExponent = maxExponent + maxPrecision - 1;
  const unsigned int maxPowerOfFiveParts = 2 + ((maxPowerOfFiveExponent * 815)
                                                / (351 * integerPartWidth));
}

/* Put a bunch of private, handy routines in an anonymous namespace.  */
namespace {

  inline unsigned int
  partCountForBits(unsigned int bits)
  {
    return ((bits) + integerPartWidth - 1) / integerPartWidth;
  }

  /* Returns 0U-9U.  Return values >= 10U are not digits.  */
  inline unsigned int
  decDigitValue(unsigned int c)
  {
    return c - '0';
  }

  unsigned int
  hexDigitValue(unsigned int c)
  {
    unsigned int r;

    r = c - '0';
    if(r <= 9)
      return r;

    r = c - 'A';
    if(r <= 5)
      return r + 10;

    r = c - 'a';
    if(r <= 5)
      return r + 10;

    return -1U;
  }

    // Return value of readExponent.
    enum readExponentStatus {
        resOK,
        resUnderflow,
        resOverflow
    };

  /* Read a decimal exponent of the form [+-]ddddddd, add delta, and place the
     result in exponent.

     Returns a value of type readExponentStatus.  */
  enum readExponentStatus
  readExponent(const char *p, int delta, exponent_t &exponent)
  {
    bool isNegative, overflow;
    integerPart absExponent;
    unsigned int value;

    overflow = false;
    isNegative = (*p == '-');
    if (*p == '-' || *p == '+')
      p++;

    value = decDigitValue(*p++);
    assert (value < 10U);
    absExponent = value;

    for (;;) {
      value = decDigitValue(*p);
      if (value >= 10U)
        break;

      p++;
      if (absExponent > ((integerPart) -1 - value) / 10) {
          overflow = true;
          break;
      }
      absExponent = absExponent * 10 + value;
    }

    // Add delta to the exponent but in absolute-value terms.  Check if the delta and the
    // exponent have the same sign.
    // Doing it like this is tricky but ensures we correctly handle the widest range of
    // exponent and delta values without prematurely overflowing.
    if (isNegative == (delta < 0)) {
        if (absExponent + abs(delta) < absExponent)
            overflow = true;
        else
            absExponent += abs(delta);
    } else {
        if (absExponent >= abs(delta))
            absExponent -= abs(delta);
        else {
            absExponent = abs(delta) - absExponent;
            isNegative = not isNegative;
        }
    }

    if (absExponent > maxAbsExponent)
        overflow = true;

    if (overflow) {
        /* Set the exponent to maximum or minimum; this ensures hexadecimal floating point
           reading normalizes appropriately.  */
        if (isNegative) {
            exponent = -maxAbsExponent;
            return resUnderflow;
        } else {
            exponent = maxAbsExponent;
            return resOverflow;
        }
    }

    exponent = (int) absExponent;
    assert(exponent == absExponent);
    if (isNegative)
        exponent = -exponent;

    return resOK;
  }

  const char *
  skipLeadingZeroesAndAnyDot(const char *p, const char **dot)
  {
    *dot = 0;
    while(*p == '0')
      p++;

    if(*p == '.') {
      *dot = p++;
      while(*p == '0')
        p++;
    }

    return p;
  }

  /* Given a normal decimal floating point number of the form

       dddd.dddd[eE][+-]ddd

     where the decimal point and exponent are optional, fill out the
     structure D.  Exponent is appropriate if the significand is
     treated as an integer, and normalizedExponent if the significand
     is taken to have the decimal point after a single leading
     non-zero digit.

     If the value is zero, V->firstSigDigit points to a non-digit, and
     the return exponent is zero.
  */
  struct decimalInfo {
    const char *firstSigDigit;
    const char *lastSigDigit;
    int exponent;
    int normalizedExponent;
    enum readExponentStatus res;
  };

  void
  interpretDecimal(const char *p, decimalInfo *D)
  {
    const char *dot;

    p = skipLeadingZeroesAndAnyDot (p, &dot);

    D->firstSigDigit = p;
    D->exponent = 0;
    D->normalizedExponent = 0;
    D->res = resOK;

    for (;;) {
      if (*p == '.') {
        assert(dot == 0);
        dot = p++;
      }
      if (decDigitValue(*p) >= 10U)
        break;
      p++;
    }

    /* If number is all zerooes accept any exponent.  */
    if (decDigitValue(*p) >= 10U) {
      if (*p == 'e' || *p == 'E') {
          exponent_t exponent;
          D->res = readExponent(p + 1, 0, exponent);
          D->exponent = exponent;
      }

      /* Implied decimal point?  */
      if (!dot)
        dot = p;

      /* Drop insignificant trailing zeroes.  */
      do
        do
          p--;
        while (*p == '0');
      while (*p == '.');

      /* Adjust the exponents for any decimal point.  */
      D->exponent += (dot - p) - (dot > p);
      D->normalizedExponent = (D->exponent + (p - D->firstSigDigit)
                               - (dot > D->firstSigDigit && dot < p));
    }

    D->lastSigDigit = p;
  }

  /* Return the trailing fraction of a hexadecimal number.
     DIGITVALUE is the first hex digit of the fraction, P points to
     the next digit.  */
  lostFraction
  trailingHexadecimalFraction(const char *p, unsigned int digitValue)
  {
    unsigned int hexDigit;

    /* If the first trailing digit isn't 0 or 8 we can work out the
       fraction immediately.  */
    if(digitValue > 8)
      return lfMoreThanHalf;
    else if(digitValue < 8 && digitValue > 0)
      return lfLessThanHalf;

    /* Otherwise we need to find the first non-zero digit skipping any dot.  */
    while(*p == '0' || *p == '.')
      p++;

    hexDigit = hexDigitValue(*p);

    /* If we ran off the end it is exactly zero or one-half, otherwise
       a little more.  */
    if(hexDigit == -1U)
      return digitValue == 0 ? lfExactlyZero: lfExactlyHalf;
    else
      return digitValue == 0 ? lfLessThanHalf: lfMoreThanHalf;
  }

  /* Return the fraction lost were a bignum truncated losing the least
     significant BITS bits.  */
  lostFraction
  lostFractionThroughTruncation(const integerPart *parts,
                                unsigned int partCount,
                                unsigned int bits)
  {
    unsigned int lsb;

    lsb = APInt::tcLSB(parts, partCount);

    /* Note this is guaranteed true if bits == 0, or LSB == -1U.  */
    if(bits <= lsb)
      return lfExactlyZero;
    if(bits == lsb + 1)
      return lfExactlyHalf;
    if(bits <= partCount * integerPartWidth
       && APInt::tcExtractBit(parts, bits - 1))
      return lfMoreThanHalf;

    return lfLessThanHalf;
  }

  /* Shift DST right BITS bits noting lost fraction.  */
  lostFraction
  shiftRight(integerPart *dst, unsigned int parts, unsigned int bits)
  {
    lostFraction lost_fraction;

    lost_fraction = lostFractionThroughTruncation(dst, parts, bits);

    APInt::tcShiftRight(dst, parts, bits);

    return lost_fraction;
  }

  /* Combine the effect of two lost fractions.  */
  lostFraction
  combineLostFractions(lostFraction moreSignificant,
                       lostFraction lessSignificant)
  {
    if(lessSignificant != lfExactlyZero) {
      if(moreSignificant == lfExactlyZero)
        moreSignificant = lfLessThanHalf;
      else if(moreSignificant == lfExactlyHalf)
        moreSignificant = lfMoreThanHalf;
    }

    return moreSignificant;
  }

  /* The error from the true value, in half-ulps, on multiplying two
     floating point numbers, which differ from the value they
     approximate by at most HUE1 and HUE2 half-ulps, is strictly less
     than the returned value.

     See "How to Read Floating Point Numbers Accurately" by William D
     Clinger.  */
  unsigned int
  HUerrBound(bool inexactMultiply, unsigned int HUerr1, unsigned int HUerr2)
  {
    assert(HUerr1 < 2 || HUerr2 < 2 || (HUerr1 + HUerr2 < 8));

    if (HUerr1 + HUerr2 == 0)
      return inexactMultiply * 2;  /* <= inexactMultiply half-ulps.  */
    else
      return inexactMultiply + 2 * (HUerr1 + HUerr2);
  }

  /* The number of ulps from the boundary (zero, or half if ISNEAREST)
     when the least significant BITS are truncated.  BITS cannot be
     zero.  */
  integerPart
  ulpsFromBoundary(const integerPart *parts, unsigned int bits, bool isNearest)
  {
    unsigned int count, partBits;
    integerPart part, boundary;

    assert (bits != 0);

    bits--;
    count = bits / integerPartWidth;
    partBits = bits % integerPartWidth + 1;

    part = parts[count] & (~(integerPart) 0 >> (integerPartWidth - partBits));

    if (isNearest)
      boundary = (integerPart) 1 << (partBits - 1);
    else
      boundary = 0;

    if (count == 0) {
      if (part - boundary <= boundary - part)
        return part - boundary;
      else
        return boundary - part;
    }

    if (part == boundary) {
      while (--count)
        if (parts[count])
          return ~(integerPart) 0; /* A lot.  */

      return parts[0];
    } else if (part == boundary - 1) {
      while (--count)
        if (~parts[count])
          return ~(integerPart) 0; /* A lot.  */

      return -parts[0];
    }

    return ~(integerPart) 0; /* A lot.  */
  }


  /* Place pow(5, power) in DST, and return the number of parts used.
     DST must be at least one part larger than size of the answer.  */
  unsigned int
  powerOf5(integerPart *dst, unsigned int power)
  {
    static integerPart firstEightPowers[] = { 1, 5, 25, 125, 625, 3125,
                                              15625, 78125 };
    static integerPart pow5s[maxPowerOfFiveParts * 2 + 5] = { 78125 * 5 };
    static unsigned int partsCount[16] = { 1 };

    integerPart scratch[maxPowerOfFiveParts], *p1, *p2, *pow5;
    unsigned int result;

    assert(power <= maxExponent);

    p1 = dst;
    p2 = scratch;

    *p1 = firstEightPowers[power & 7];
    power >>= 3;

    result = 1;
    pow5 = pow5s;

    for (unsigned int n = 0; power; power >>= 1, n++) {
      unsigned int pc;

      pc = partsCount[n];

      /* Calculate pow(5,pow(2,n+3)) if we haven't yet.  */
      if (pc == 0) {
        pc = partsCount[n - 1];
        pc = APInt::tcFullMultiply(pow5, pow5 - pc, pow5 - pc, pc, pc);
        partsCount[n] = pc;
      }

      if (power & 1) {
        result = APInt::tcFullMultiply(p2, p1, pow5, result, pc);

        /* Now result is in p1 with partsCount parts and p2 is scratch
           space.  */
        integerPart *tmp;
        tmp = p1, p1 = p2, p2 = tmp;
      }

      pow5 += pc;
    }

    if (p1 != dst)
      APInt::tcAssign(dst, p1, result);

    return result;
  }

  /* Zero at the end to avoid modular arithmetic when adding one; used
     when rounding up during hexadecimal output.  */
  static const char hexDigitsLower[] = "0123456789abcdef0";
  static const char hexDigitsUpper[] = "0123456789ABCDEF0";
  static const char infinityL[] = "infinity";
  static const char infinityU[] = "INFINITY";
  static const char NaNL[] = "nan";
  static const char NaNU[] = "NAN";

  /* Write out an integerPart in hexadecimal, starting with the most
     significant nibble.  Write out exactly COUNT hexdigits, return
     COUNT.  */
  unsigned int
  partAsHex (char *dst, integerPart part, unsigned int count,
             const char *hexDigitChars)
  {
    unsigned int result = count;

    assert (count != 0 && count <= integerPartWidth / 4);

    part >>= (integerPartWidth - 4 * count);
    while (count--) {
      dst[count] = hexDigitChars[part & 0xf];
      part >>= 4;
    }

    return result;
  }

  /* Write out an unsigned decimal integer.  */
  char *
  writeUnsignedDecimal (char *dst, unsigned int n)
  {
    char buff[40], *p;

    p = buff;
    do
      *p++ = '0' + n % 10;
    while (n /= 10);

    do
      *dst++ = *--p;
    while (p != buff);

    return dst;
  }

  /* Write out a signed decimal integer.  */
  char *
  writeSignedDecimal (char *dst, int value)
  {
    if (value < 0) {
      *dst++ = '-';
      dst = writeUnsignedDecimal(dst, -(unsigned) value);
    } else
      dst = writeUnsignedDecimal(dst, value);

    return dst;
  }
}

/* Constructors.  */
void
APFloat::initialize(const fltSemantics *ourSemantics)
{
  unsigned int count;

  semantics = ourSemantics;
  count = partCount();
  if(count > 1)
    significand.parts = new integerPart[count];
}

void
APFloat::freeSignificand()
{
  if(partCount() > 1)
    delete [] significand.parts;
}

void
APFloat::assign(const APFloat &rhs)
{
  assert(semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if(category == fcNormal)
    copySignificand(rhs);
}

void
APFloat::copySignificand(const APFloat &rhs)
{
  assert(category == fcNormal);
  assert(rhs.partCount() >= partCount());

  APInt::tcAssign(significandParts(), rhs.significandParts(),
                  partCount());
}

APFloat &
APFloat::operator=(const APFloat &rhs)
{
  if(this != &rhs) {
    if(semantics != rhs.semantics) {
      freeSignificand();
      initialize(rhs.semantics);
    }
    assign(rhs);
  }

  return *this;
}

APFloat::APFloat(const fltSemantics &ourSemantics, integerPart value)
{
  initialize(&ourSemantics);
  sign = 0;
  zeroSignificand();
  exponent = ourSemantics.precision - 1;
  significandParts()[0] = value;
  normalize(rmNearestTiesToEven, lfExactlyZero);
}

APFloat::APFloat(const fltSemantics &ourSemantics,
                 fltCategory ourCategory, bool negative)
{
  initialize(&ourSemantics);
  category = ourCategory;
  sign = negative;
  if(category == fcNormal)
    category = fcZero;
}

APFloat::APFloat(const fltSemantics &ourSemantics, const char *text)
{
  initialize(&ourSemantics);
  convertFromString(text, rmNearestTiesToEven);
}

APFloat::APFloat(const APFloat &rhs)
{
  initialize(rhs.semantics);
  assign(rhs);
}

APFloat::~APFloat()
{
  freeSignificand();
}

unsigned int
APFloat::partCount() const
{
  return partCountForBits(semantics->precision + 1);
}

unsigned int
APFloat::semanticsPrecision(const fltSemantics &semantics)
{
  return semantics.precision;
}

const integerPart *
APFloat::significandParts() const
{
  return const_cast<APFloat *>(this)->significandParts();
}

integerPart *
APFloat::significandParts()
{
  assert(category == fcNormal);

  if(partCount() > 1)
    return significand.parts;
  else
    return &significand.part;
}

void
APFloat::zeroSignificand()
{
  category = fcNormal;
  APInt::tcSet(significandParts(), 0, partCount());
}

/* Increment an fcNormal floating point number's significand.  */
void
APFloat::incrementSignificand()
{
  integerPart carry;

  carry = APInt::tcIncrement(significandParts(), partCount());

  /* Our callers should never cause us to overflow.  */
  assert(carry == 0);
}

/* Add the significand of the RHS.  Returns the carry flag.  */
integerPart
APFloat::addSignificand(const APFloat &rhs)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcAdd(parts, rhs.significandParts(), 0, partCount());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
integerPart
APFloat::subtractSignificand(const APFloat &rhs, integerPart borrow)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcSubtract(parts, rhs.significandParts(), borrow,
                           partCount());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
lostFraction
APFloat::multiplySignificand(const APFloat &rhs, const APFloat *addend)
{
  unsigned int omsb;    // One, not zero, based MSB.
  unsigned int partsCount, newPartsCount, precision, extendedPrecision;
  integerPart *lhsSignificand;
  integerPart scratch[4];
  integerPart *fullSignificand;
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  precision = semantics->precision;
  extendedPrecision = precision * 2;
  newPartsCount = partCountForBits(extendedPrecision);

  if(newPartsCount > 4)
    fullSignificand = new integerPart[newPartsCount];
  else
    fullSignificand = scratch;

  lhsSignificand = significandParts();
  partsCount = partCount();

  APInt::tcFullMultiply(fullSignificand, lhsSignificand,
                        rhs.significandParts(), partsCount, partsCount);

  lost_fraction = lfExactlyZero;
  omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;

  /* Below we adjust the exponent to the extended precision.  The multiplied values are

       sig1 * 2 ^ (exp1 + (precision - 1))
       sig2 * 2 ^ (exp2 * (precision - 1))

     So writing the product as:

       sig1 * sig2 * 2 ^ ((exp1 + exp2 + 1) + (extendedPrecision - 1))

     we have the multiplied significand with exponent (exp1 + exp2 + 1) in extendedPrecision.
  */
  if(addend && addend->category == fcNormal) {
    opStatus status;

    /* Create new full-precision semantics and convert addend to those semantics.  */
    fltSemantics extendedSemantics(*semantics);
    extendedSemantics.precision = extendedPrecision;

    APFloat extendedAddend(*addend);
    status = extendedAddend.convert(extendedSemantics, rmTowardZero);
    assert(status == opOK);

    // Only modify ourselves after copying addend, in case addend is this object
    exponent += rhs.exponent + 1;

    /* Normalize our MSB.  */
    if(omsb != extendedPrecision)
      {
        APInt::tcShiftLeft(fullSignificand, newPartsCount, extendedPrecision - omsb);
        exponent -= extendedPrecision - omsb;
      }

    /* Perform the addition by pretending our objbect's memory is fullSignificand.  */
    Significand savedSignificand = significand;
    const fltSemantics *savedSemantics = semantics;

    if(newPartsCount == 1)
      significand.part = fullSignificand[0];
    else
      significand.parts = fullSignificand;
    semantics = &extendedSemantics;

    lost_fraction = addOrSubtractSignificand(extendedAddend, false);

    /* Go back to using our original storage and semantics; ensuring the full precision
       result is in fullSignificand.  */
    if(newPartsCount == 1)
      fullSignificand[0] = significand.part;
    significand = savedSignificand;
    semantics = savedSemantics;

    omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  }
  else
    exponent += rhs.exponent + 1;

  /* Now narrow from the extended precision back to normal precision, shifting the
     significand if necessary.  */
  exponent -= precision;

  if(omsb > precision) {
    unsigned int bits, significantParts;
    lostFraction lf;

    bits = omsb - precision;
    significantParts = partCountForBits(omsb);
    lf = shiftRight(fullSignificand, significantParts, bits);
    lost_fraction = combineLostFractions(lf, lost_fraction);
    exponent += bits;
  }

  APInt::tcAssign(lhsSignificand, fullSignificand, partsCount);

  if(newPartsCount > 4)
    delete [] fullSignificand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
lostFraction
APFloat::divideSignificand(const APFloat &rhs)
{
  unsigned int bit, i, partsCount;
  const integerPart *rhsSignificand;
  integerPart *lhsSignificand, *dividend, *divisor;
  integerPart scratch[4];
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  lhsSignificand = significandParts();
  rhsSignificand = rhs.significandParts();
  partsCount = partCount();

  if(partsCount > 2)
    dividend = new integerPart[partsCount * 2];
  else
    dividend = scratch;

  divisor = dividend + partsCount;

  /* Copy the dividend and divisor as they will be modified in-place.  */
  for(i = 0; i < partsCount; i++) {
    dividend[i] = lhsSignificand[i];
    divisor[i] = rhsSignificand[i];
    lhsSignificand[i] = 0;
  }

  exponent -= rhs.exponent;

  unsigned int precision = semantics->precision;

  /* Normalize the divisor.  */
  bit = precision - APInt::tcMSB(divisor, partsCount) - 1;
  if(bit) {
    exponent += bit;
    APInt::tcShiftLeft(divisor, partsCount, bit);
  }

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB(dividend, partsCount) - 1;
  if(bit) {
    exponent -= bit;
    APInt::tcShiftLeft(dividend, partsCount, bit);
  }

  /* Ensure the dividend >= divisor initially for the loop below.
     Incidentally, this means that the division loop below is
     guaranteed to set the integer bit to one.  */
  if(APInt::tcCompare(dividend, divisor, partsCount) < 0) {
    exponent--;
    APInt::tcShiftLeft(dividend, partsCount, 1);
    assert(APInt::tcCompare(dividend, divisor, partsCount) >= 0);
  }

  /* Long division.  */
  for(bit = precision; bit; bit -= 1) {
    if(APInt::tcCompare(dividend, divisor, partsCount) >= 0) {
      APInt::tcSubtract(dividend, divisor, 0, partsCount);
      APInt::tcSetBit(lhsSignificand, bit - 1);
    }

    APInt::tcShiftLeft(dividend, partsCount, 1);
  }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tcCompare(dividend, divisor, partsCount);

  if(cmp > 0)
    lost_fraction = lfMoreThanHalf;
  else if(cmp == 0)
    lost_fraction = lfExactlyHalf;
  else if(APInt::tcIsZero(dividend, partsCount))
    lost_fraction = lfExactlyZero;
  else
    lost_fraction = lfLessThanHalf;

  if(partsCount > 2)
    delete [] dividend;

  return lost_fraction;
}

unsigned int
APFloat::significandMSB() const
{
  return APInt::tcMSB(significandParts(), partCount());
}

unsigned int
APFloat::significandLSB() const
{
  return APInt::tcLSB(significandParts(), partCount());
}

/* Note that a zero result is NOT normalized to fcZero.  */
lostFraction
APFloat::shiftSignificandRight(unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return shiftRight(significandParts(), partCount(), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
APFloat::shiftSignificandLeft(unsigned int bits)
{
  assert(bits < semantics->precision);

  if(bits) {
    unsigned int partsCount = partCount();

    APInt::tcShiftLeft(significandParts(), partsCount, bits);
    exponent -= bits;

    assert(!APInt::tcIsZero(significandParts(), partsCount));
  }
}

APFloat::cmpResult
APFloat::compareAbsoluteValue(const APFloat &rhs) const
{
  int compare;

  assert(semantics == rhs.semantics);
  assert(category == fcNormal);
  assert(rhs.category == fcNormal);

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if(compare == 0)
    compare = APInt::tcCompare(significandParts(), rhs.significandParts(),
                               partCount());

  if(compare > 0)
    return cmpGreaterThan;
  else if(compare < 0)
    return cmpLessThan;
  else
    return cmpEqual;
}

/* Handle overflow.  Sign is preserved.  We either become infinity or
   the largest finite number.  */
APFloat::opStatus
APFloat::handleOverflow(roundingMode rounding_mode)
{
  /* Infinity?  */
  if(rounding_mode == rmNearestTiesToEven
     || rounding_mode == rmNearestTiesToAway
     || (rounding_mode == rmTowardPositive && !sign)
     || (rounding_mode == rmTowardNegative && sign))
    {
      category = fcInfinity;
      return (opStatus) (opOverflow | opInexact);
    }

  /* Otherwise we become the largest finite number.  */
  category = fcNormal;
  exponent = semantics->maxExponent;
  APInt::tcSetLeastSignificantBits(significandParts(), partCount(),
                                   semantics->precision);

  return opInexact;
}

/* Returns TRUE if, when truncating the current number, with BIT the
   new LSB, with the given lost fraction and rounding mode, the result
   would need to be rounded away from zero (i.e., by increasing the
   signficand).  This routine must work for fcZero of both signs, and
   fcNormal numbers.  */
bool
APFloat::roundAwayFromZero(roundingMode rounding_mode,
                           lostFraction lost_fraction,
                           unsigned int bit) const
{
  /* NaNs and infinities should not have lost fractions.  */
  assert(category == fcNormal || category == fcZero);

  /* Current callers never pass this so we don't handle it.  */
  assert(lost_fraction != lfExactlyZero);

  switch(rounding_mode) {
  default:
    assert(0);

  case rmNearestTiesToAway:
    return lost_fraction == lfExactlyHalf || lost_fraction == lfMoreThanHalf;

  case rmNearestTiesToEven:
    if(lost_fraction == lfMoreThanHalf)
      return true;

    /* Our zeroes don't have a significand to test.  */
    if(lost_fraction == lfExactlyHalf && category != fcZero)
      return APInt::tcExtractBit(significandParts(), bit);

    return false;

  case rmTowardZero:
    return false;

  case rmTowardPositive:
    return sign == false;

  case rmTowardNegative:
    return sign == true;
  }
}

APFloat::opStatus
APFloat::normalize(roundingMode rounding_mode,
                   lostFraction lost_fraction)
{
  unsigned int omsb;            /* One, not zero, based MSB.  */
  int exponentChange;

  if(category != fcNormal)
    return opOK;

  /* Before rounding normalize the exponent of fcNormal numbers.  */
  omsb = significandMSB() + 1;

  if(omsb) {
    /* OMSB is numbered from 1.  We want to place it in the integer
       bit numbered PRECISION if possible, with a compensating change in
       the exponent.  */
    exponentChange = omsb - semantics->precision;

    /* If the resulting exponent is too high, overflow according to
       the rounding mode.  */
    if(exponent + exponentChange > semantics->maxExponent)
      return handleOverflow(rounding_mode);

    /* Subnormal numbers have exponent minExponent, and their MSB
       is forced based on that.  */
    if(exponent + exponentChange < semantics->minExponent)
      exponentChange = semantics->minExponent - exponent;

    /* Shifting left is easy as we don't lose precision.  */
    if(exponentChange < 0) {
      assert(lost_fraction == lfExactlyZero);

      shiftSignificandLeft(-exponentChange);

      return opOK;
    }

    if(exponentChange > 0) {
      lostFraction lf;

      /* Shift right and capture any new lost fraction.  */
      lf = shiftSignificandRight(exponentChange);

      lost_fraction = combineLostFractions(lf, lost_fraction);

      /* Keep OMSB up-to-date.  */
      if(omsb > (unsigned) exponentChange)
        omsb -= exponentChange;
      else
        omsb = 0;
    }
  }

  /* Now round the number according to rounding_mode given the lost
     fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if(lost_fraction == lfExactlyZero) {
    /* Canonicalize zeroes.  */
    if(omsb == 0)
      category = fcZero;

    return opOK;
  }

  /* Increment the significand if we're rounding away from zero.  */
  if(roundAwayFromZero(rounding_mode, lost_fraction, 0)) {
    if(omsb == 0)
      exponent = semantics->minExponent;

    incrementSignificand();
    omsb = significandMSB() + 1;

    /* Did the significand increment overflow?  */
    if(omsb == (unsigned) semantics->precision + 1) {
      /* Renormalize by incrementing the exponent and shifting our
         significand right one.  However if we already have the
         maximum exponent we overflow to infinity.  */
      if(exponent == semantics->maxExponent) {
        category = fcInfinity;

        return (opStatus) (opOverflow | opInexact);
      }

      shiftSignificandRight(1);

      return opInexact;
    }
  }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if(omsb == semantics->precision)
    return opInexact;

  /* We have a non-zero denormal.  */
  assert(omsb < semantics->precision);

  /* Canonicalize zeroes.  */
  if(omsb == 0)
    category = fcZero;

  /* The fcZero case is a denormal that underflowed to zero.  */
  return (opStatus) (opUnderflow | opInexact);
}

APFloat::opStatus
APFloat::addOrSubtractSpecials(const APFloat &rhs, bool subtract)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcNormal, fcZero):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcZero):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    return opOK;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcZero, fcInfinity):
    category = fcInfinity;
    sign = rhs.sign ^ subtract;
    return opOK;

  case convolve(fcZero, fcNormal):
    assign(rhs);
    sign = rhs.sign ^ subtract;
    return opOK;

  case convolve(fcZero, fcZero):
    /* Sign depends on rounding mode; handled by caller.  */
    return opOK;

  case convolve(fcInfinity, fcInfinity):
    /* Differently signed infinities can only be validly
       subtracted.  */
    if((sign ^ rhs.sign) != subtract) {
      category = fcNaN;
      return opInvalidOp;
    }

    return opOK;

  case convolve(fcNormal, fcNormal):
    return opDivByZero;
  }
}

/* Add or subtract two normal numbers.  */
lostFraction
APFloat::addOrSubtractSignificand(const APFloat &rhs, bool subtract)
{
  integerPart carry;
  lostFraction lost_fraction;
  int bits;

  assert(category == fcNormal);
  assert(rhs.category == fcNormal);

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= (sign ^ rhs.sign);

  /* Are we bigger exponent-wise than the RHS?  */
  bits = exponent - rhs.exponent;

  /* Subtraction is more subtle than one might naively expect.  */
  if(subtract) {
    APFloat temp_rhs(rhs);

    if (bits == 0) {
      lost_fraction = lfExactlyZero;
    } else if (bits > 0) {
      lost_fraction = temp_rhs.shiftSignificandRight(bits - 1);
      shiftSignificandLeft(1);
    } else {
      lost_fraction = shiftSignificandRight(-bits - 1);
      temp_rhs.shiftSignificandLeft(1);
    }

    if (compareAbsoluteValue(temp_rhs) == cmpLessThan) {
      carry = temp_rhs.subtractSignificand
        (*this, lost_fraction != lfExactlyZero);
      copySignificand(temp_rhs);
      sign = !sign;
    } else {
      carry = subtractSignificand
        (temp_rhs, lost_fraction != lfExactlyZero);
    }

    /* Invert the lost fraction - it was on the RHS and
       subtracted.  */
    if(lost_fraction == lfLessThanHalf)
      lost_fraction = lfMoreThanHalf;
    else if(lost_fraction == lfMoreThanHalf)
      lost_fraction = lfLessThanHalf;

    /* The code above is intended to ensure that no borrow is
       necessary.  */
    assert(!carry);
  } else {
    if(bits > 0) {
      APFloat temp_rhs(rhs);

      lost_fraction = temp_rhs.shiftSignificandRight(bits);
      carry = addSignificand(temp_rhs);
    } else {
      lost_fraction = shiftSignificandRight(-bits);
      carry = addSignificand(rhs);
    }

    /* We have a guard bit; generating a carry cannot happen.  */
    assert(!carry);
  }

  return lost_fraction;
}

APFloat::opStatus
APFloat::multiplySpecials(const APFloat &rhs)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    return opOK;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcInfinity):
    category = fcInfinity;
    return opOK;

  case convolve(fcZero, fcNormal):
  case convolve(fcNormal, fcZero):
  case convolve(fcZero, fcZero):
    category = fcZero;
    return opOK;

  case convolve(fcZero, fcInfinity):
  case convolve(fcInfinity, fcZero):
    category = fcNaN;
    return opInvalidOp;

  case convolve(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus
APFloat::divideSpecials(const APFloat &rhs)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcInfinity, fcZero):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcZero, fcInfinity):
  case convolve(fcZero, fcNormal):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    return opOK;

  case convolve(fcNormal, fcInfinity):
    category = fcZero;
    return opOK;

  case convolve(fcNormal, fcZero):
    category = fcInfinity;
    return opDivByZero;

  case convolve(fcInfinity, fcInfinity):
  case convolve(fcZero, fcZero):
    category = fcNaN;
    return opInvalidOp;

  case convolve(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus
APFloat::modSpecials(const APFloat &rhs)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case convolve(fcInfinity, fcZero):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcInfinity):
  case convolve(fcZero, fcZero):
  case convolve(fcNormal, fcZero):
    category = fcNaN;
    return opInvalidOp;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcZero, fcInfinity):
  case convolve(fcZero, fcNormal):
    /* We retain our sign, per IEEE754.  */
    category = fcZero;
    return opOK;

  case convolve(fcNormal, fcNormal):
    return opOK;
  }
}

/* Change sign.  */
void
APFloat::changeSign()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

void
APFloat::clearSign()
{
  /* So is this one. */
  sign = 0;
}

void
APFloat::copySign(const APFloat &rhs)
{
  /* And this one. */
  sign = rhs.sign;
}

/* Normalized addition or subtraction.  */
APFloat::opStatus
APFloat::addOrSubtract(const APFloat &rhs, roundingMode rounding_mode,
                       bool subtract)
{
  opStatus fs;

  fs = addOrSubtractSpecials(rhs, subtract);

  /* This return code means it was not a simple case.  */
  if(fs == opDivByZero) {
    lostFraction lost_fraction;

    lost_fraction = addOrSubtractSignificand(rhs, subtract);
    fs = normalize(rounding_mode, lost_fraction);

    /* Can only be zero if we lost no fraction.  */
    assert(category != fcZero || lost_fraction == lfExactlyZero);
  }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if(category == fcZero) {
    if(rhs.category != fcZero || (sign == rhs.sign) == subtract)
      sign = (rounding_mode == rmTowardNegative);
  }

  return fs;
}

/* Normalized addition.  */
APFloat::opStatus
APFloat::add(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
APFloat::opStatus
APFloat::subtract(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, true);
}

/* Normalized multiply.  */
APFloat::opStatus
APFloat::multiply(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = multiplySpecials(rhs);

  if(category == fcNormal) {
    lostFraction lost_fraction = multiplySignificand(rhs, NULL);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized divide.  */
APFloat::opStatus
APFloat::divide(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = divideSpecials(rhs);

  if(category == fcNormal) {
    lostFraction lost_fraction = divideSignificand(rhs);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Common code for C90/C99 and IEEE remainder operations.  */
APFloat
APFloat::remQuo(const APFloat &rhs, bool is_ieee)
{
  APFloat dividendF (*this);
  APFloat divisorF (rhs);

  assert(semantics == rhs.semantics);

  integerPart *divisor = divisorF.significandParts();
  integerPart *dividend = dividendF.significandParts();
  unsigned int precision = semantics->precision;
  unsigned int partsCount = partCount();

  /* Normalize the divisor.  */
  unsigned int bit;
  bit = precision - APInt::tcMSB(divisor, partsCount) - 1;
  divisorF.shiftSignificandLeft (bit);

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB(dividend, partsCount) - 1;
  dividendF.shiftSignificandLeft (bit);

  /* Ensure the dividend is greater than the divisor.  */
  if(APInt::tcCompare(dividend, divisor, partsCount) < 0) {
    dividendF.shiftSignificandLeft (1);
    assert(APInt::tcCompare(dividend, divisor, partsCount) >= 0);
  }

  /* We are the quotient.  */
  integerPart *quotient = significandParts();
  zeroSignificand();

  bool ieee_rounded_quotient = false;

  exponent = dividendF.exponent - divisorF.exponent;

  if (exponent >= -1) {
    /* Just enough long division to get an integer.  */
    bit = exponent + 1;
    if (bit > precision)
      bit = precision;

    dividendF.exponent = divisorF.exponent;

    for(; bit; bit -= 1) {
      if(APInt::tcCompare(dividend, divisor, partsCount) >= 0) {
        APInt::tcSubtract(dividend, divisor, 0, partsCount);
        APInt::tcSetBit(quotient, bit - 1);
      }

      APInt::tcShiftLeft(dividend, partsCount, 1);
    }

    /* The quotient is an integer placed in LSBs.  */
    exponent += precision - 1;

    /* IEEE requires the quotient to be rounded-to-nearest.  */
    if (is_ieee) {
      int cmp = APInt::tcCompare(dividend, divisor, partsCount);

      if(cmp > 0 || (cmp == 0 && (quotient[0] & 1))) {
        incrementSignificand();
        ieee_rounded_quotient = true;
      }
    }
  }

  /* Normalize.  */
  opStatus fs;
  fs = normalize(rmTowardZero, lfExactlyZero);
  assert (fs == opOK);

  /* If this gives zero our sign is unchanged, as IEEE requires.  */
  fs = dividendF.normalize(rmTowardZero, lfExactlyZero);
  assert (fs == opOK);

  if (ieee_rounded_quotient) {
    lostFraction lost_fraction;

    lost_fraction = dividendF.addOrSubtractSignificand(rhs, !sign);
    assert(lost_fraction == lfExactlyZero);
    fs = dividendF.normalize(rmTowardZero, lost_fraction);
    assert(dividendF.category != fcZero && fs == opOK);
  }

  return dividendF;
}

/* C90/C99 fmod remainder.  */
APFloat::opStatus
APFloat::fmod(const APFloat &rhs)
{
  opStatus fs;

  fs = modSpecials(rhs);

  if (category == fcNormal)
    *this = remQuo (rhs, false);

  return fs;
}

/* C99/IEEE-754 remainder.  */
APFloat::opStatus
APFloat::remainder(const APFloat &rhs)
{
  opStatus fs;

  fs = modSpecials(rhs);

  if (category == fcNormal)
    *this = remQuo (rhs, true);

  return fs;
}

/* Normalized fused-multiply-add.  */
APFloat::opStatus
APFloat::fusedMultiplyAdd(const APFloat &multiplicand,
                          const APFloat &addend,
                          roundingMode rounding_mode)
{
  opStatus fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if(category == fcNormal
     && multiplicand.category == fcNormal
     && (addend.category == fcNormal || addend.category == fcZero)) {
    lostFraction lost_fraction;

    lost_fraction = multiplySignificand(multiplicand, &addend);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);

    /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
       positive zero unless rounding to minus infinity, except that
       adding two like-signed zeroes gives that zero.  */
    if(category == fcZero && lost_fraction == lfExactlyZero && sign != addend.sign)
      sign = (rounding_mode == rmTowardNegative);
  } else {
    fs = multiplySpecials(multiplicand);

    /* FS can only be opOK or opInvalidOp.  There is no more work
       to do in the latter case.  The IEEE-754R standard says it is
       implementation-defined in this case whether, if ADDEND is a
       quiet NaN, we raise invalid op; this implementation does so.

       If we need to do the addition we can do so with normal
       precision.  */
    if(fs == opOK)
      fs = addOrSubtract(addend, rounding_mode, false);
  }

  return fs;
}

/* Comparison requires normalized numbers.  */
APFloat::cmpResult
APFloat::compare(const APFloat &rhs) const
{
  cmpResult result;

  assert(semantics == rhs.semantics);

  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    return cmpUnordered;

  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcZero):
  case convolve(fcNormal, fcZero):
    if(sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcZero, fcInfinity):
  case convolve(fcZero, fcNormal):
    if(rhs.sign)
      return cmpGreaterThan;
    else
      return cmpLessThan;

  case convolve(fcInfinity, fcInfinity):
    if(sign == rhs.sign)
      return cmpEqual;
    else if(sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case convolve(fcZero, fcZero):
    return cmpEqual;

  case convolve(fcNormal, fcNormal):
    break;
  }

  /* Two normal numbers.  Do they have the same sign?  */
  if(sign != rhs.sign) {
    if(sign)
      result = cmpLessThan;
    else
      result = cmpGreaterThan;
  } else {
    /* Compare absolute values; invert result if negative.  */
    result = compareAbsoluteValue(rhs);

    if(sign) {
      if(result == cmpLessThan)
        result = cmpGreaterThan;
      else if(result == cmpGreaterThan)
        result = cmpLessThan;
    }
  }

  return result;
}

APFloat::opStatus
APFloat::convert(const fltSemantics &toSemantics, roundingMode rounding_mode)
{
  lostFraction lostFraction;
  unsigned int newPartCount, oldPartCount, left_shift;
  opStatus fs;

  lostFraction = lfExactlyZero;
  newPartCount = partCountForBits(toSemantics.precision + 1);
  oldPartCount = partCount();
  left_shift = toSemantics.precision - semantics->precision;

  /* Handle storage complications.  If our new form is wider,
     re-allocate our bit pattern into wider storage.  If it is
     narrower, we ignore the excess parts, but if narrowing to a
     single part we need to free the old storage.  */
  if (newPartCount > oldPartCount) {
    integerPart *newParts;

    newParts = new integerPart[newPartCount];
    APInt::tcSet(newParts, 0, newPartCount);
    APInt::tcAssign(newParts, significandParts(), oldPartCount);
    freeSignificand();
    significand.parts = newParts;
  } else if (newPartCount < oldPartCount) {
    if(category == fcNormal) {
      /* Shift the significand right and capture any lost fraction through truncation.  */
      lostFraction = shiftRight(significandParts(), oldPartCount, -left_shift);
      exponent -= left_shift;
    }

    if (newPartCount == 1) {
      integerPart newPart = significandParts()[0];
      freeSignificand();
      significand.part = newPart;
    }
  }

  if(category == fcNormal) {
    /* Re-interpret our bit-pattern.  */
    exponent += left_shift;
    semantics = &toSemantics;
    fs = normalize(rounding_mode, lostFraction);
  } else {
    semantics = &toSemantics;
    fs = opOK;
  }

  return fs;
}

/* Convert a floating point number to an integer according to the
   rounding mode.  If the rounded integer value is out of range this
   returns an invalid operation exception and the contents of the
   destination parts are unspecified.  If the rounded value is in
   range but the floating point number is not the exact integer, the C
   standard doesn't require an inexact exception to be raised.  IEEE
   854 does require it so we do that.

   Note that for conversions to integer type the C standard requires
   round-to-zero to always be used.  */
APFloat::opStatus
APFloat::convertToInteger(integerPart *parts, unsigned int width,
                          bool isSigned,
                          roundingMode rounding_mode) const
{
  lostFraction lost_fraction;
  const integerPart *src;
  unsigned int dstPartsCount, truncatedBits;

  /* Handle the three special cases first.  */
  if(category == fcInfinity || category == fcNaN)
    return opInvalidOp;

  dstPartsCount = partCountForBits(width);

  if(category == fcZero) {
    APInt::tcSet(parts, 0, dstPartsCount);
    return opOK;
  }

  src = significandParts();

  /* Step 1: place our absolute value, with any fraction truncated, in
     the destination.  */
  if (exponent < 0) {
    /* Our absolute value is less than one; truncate everything.  */
    APInt::tcSet(parts, 0, dstPartsCount);
    truncatedBits = semantics->precision;
  } else {
    /* We want the most significant (exponent + 1) bits; the rest are
       truncated.  */
    unsigned int bits = exponent + 1U;

    /* Hopelessly large in magnitude?  */
    if (bits > width)
      return opInvalidOp;

    if (bits < semantics->precision) {
      /* We truncate (semantics->precision - bits) bits.  */
      truncatedBits = semantics->precision - bits;
      APInt::tcExtract(parts, dstPartsCount, src, bits, truncatedBits);
    } else {
      /* We want at least as many bits as are available.  */
      APInt::tcExtract(parts, dstPartsCount, src, semantics->precision, 0);
      APInt::tcShiftLeft(parts, dstPartsCount, bits - semantics->precision);
      truncatedBits = 0;
    }
  }

  /* Step 2: work out any lost fraction, and increment the absolute
     value if we would round away from zero.  */
  if (truncatedBits) {
    lost_fraction = lostFractionThroughTruncation(src, partCount(),
                                                  truncatedBits);
    if (lost_fraction != lfExactlyZero
        && roundAwayFromZero(rounding_mode, lost_fraction, truncatedBits)) {
      if (APInt::tcIncrement(parts, dstPartsCount))
        return opInvalidOp;     /* Overflow.  */
    }
  } else {
    lost_fraction = lfExactlyZero;
  }

  /* Step 3: check if we fit in the destination.  */
  unsigned int omsb = APInt::tcMSB(parts, dstPartsCount) + 1;

  if (sign) {
    if (!isSigned) {
      /* Negative numbers cannot be represented as unsigned.  */
      if (omsb != 0)
        return opInvalidOp;
    } else {
      /* It takes omsb bits to represent the unsigned integer value.
         We lose a bit for the sign, but care is needed as the
         maximally negative integer is a special case.  */
      if (omsb == width && APInt::tcLSB(parts, dstPartsCount) + 1 != omsb)
        return opInvalidOp;

      /* This case can happen because of rounding.  */
      if (omsb > width)
        return opInvalidOp;
    }

    APInt::tcNegate (parts, dstPartsCount);
  } else {
    if (omsb >= width + !isSigned)
      return opInvalidOp;
  }

  if (lost_fraction == lfExactlyZero)
    return opOK;
  else
    return opInexact;
}

/* Convert an unsigned integer SRC to a floating point number,
   rounding according to ROUNDING_MODE.  The sign of the floating
   point number is not modified.  */
APFloat::opStatus
APFloat::convertFromUnsignedParts(const integerPart *src,
                                  unsigned int srcCount,
                                  roundingMode rounding_mode)
{
  unsigned int omsb, precision, dstCount;
  integerPart *dst;
  lostFraction lost_fraction;

  category = fcNormal;
  omsb = APInt::tcMSB(src, srcCount) + 1;
  dst = significandParts();
  dstCount = partCount();
  precision = semantics->precision;

  /* We want the most significant PRECISION bits of SRC.  There may not
     be that many; extract what we can.  */
  if (precision <= omsb) {
    exponent = omsb - 1;
    lost_fraction = lostFractionThroughTruncation(src, srcCount,
                                                  omsb - precision);
    APInt::tcExtract(dst, dstCount, src, precision, omsb - precision);
  } else {
    exponent = precision - 1;
    lost_fraction = lfExactlyZero;
    APInt::tcExtract(dst, dstCount, src, omsb, 0);
  }

  return normalize(rounding_mode, lost_fraction);
}

/* Convert a two's complement integer SRC to a floating point number,
   rounding according to ROUNDING_MODE.  ISSIGNED is true if the
   integer is signed, in which case it must be sign-extended.  */
APFloat::opStatus
APFloat::convertFromSignExtendedInteger(const integerPart *src,
                                        unsigned int srcCount,
                                        bool isSigned,
                                        roundingMode rounding_mode)
{
  opStatus status;

  if (isSigned
      && APInt::tcExtractBit(src, srcCount * integerPartWidth - 1)) {
    integerPart *copy;

    /* If we're signed and negative negate a copy.  */
    sign = true;
    copy = new integerPart[srcCount];
    APInt::tcAssign(copy, src, srcCount);
    APInt::tcNegate(copy, srcCount);
    status = convertFromUnsignedParts(copy, srcCount, rounding_mode);
    delete [] copy;
  } else {
    sign = false;
    status = convertFromUnsignedParts(src, srcCount, rounding_mode);
  }

  return status;
}

APFloat::opStatus
APFloat::convertFromHexadecimalString(const char *p,
                                      roundingMode rounding_mode)
{
  lostFraction lost_fraction;
  integerPart *significand;
  unsigned int bitPos, partsCount;
  const char *dot, *firstSigDigit;
  bool calced_lf;

  zeroSignificand();
  exponent = 0;
  category = fcNormal;

  significand = significandParts();
  partsCount = partCount();
  bitPos = partsCount * integerPartWidth;

  /* Skip leading zeroes and any (hexa)decimal point.  */
  p = skipLeadingZeroesAndAnyDot(p, &dot);
  firstSigDigit = p;

  lost_fraction = lfExactlyZero;
  calced_lf = false;
  for(;;) {
    integerPart hex_value;

    if(*p == '.') {
      assert(dot == 0);
      dot = p++;
    }

    hex_value = hexDigitValue(*p);
    if(hex_value == -1U)
      break;

    p++;

    /* Store the number whilst 4-bit nibbles remain.  */
    if(bitPos) {
      bitPos -= 4;
      hex_value <<= bitPos % integerPartWidth;
      significand[bitPos / integerPartWidth] |= hex_value;
    } else if (!calced_lf) {
      lost_fraction = trailingHexadecimalFraction(p, hex_value);
      calced_lf = true;
    }
  }

  /* Hex floats require an exponent but not a hexadecimal point.  */
  assert(*p == 'p' || *p == 'P');

  /* Ignore the exponent if we are zero.  */
  if(p != firstSigDigit) {
    int expAdjustment;

    /* Implicit hexadecimal point?  */
    if(!dot)
      dot = p;

    /* Calculate the exponent adjustment implicit in the number of
       significant digits.  */
    expAdjustment = dot - firstSigDigit;
    if(expAdjustment < 0)
      expAdjustment++;
    expAdjustment = expAdjustment * 4 - 1;

    /* Adjust for writing the significand starting at the most
       significant nibble.  */
    expAdjustment += semantics->precision;
    expAdjustment -= partsCount * integerPartWidth;

    /* Adjust for the given exponent.  */
    readExponent(p + 1, expAdjustment, exponent);
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::roundSignificandWithExponent(const integerPart *decSigParts,
                                      unsigned sigPartCount, int exp,
                                      roundingMode rounding_mode)
{
  unsigned int parts, pow5PartCount;
  fltSemantics calcSemantics = { 32767, -32767, 0 };
  integerPart pow5Parts[maxPowerOfFiveParts];
  bool isNearest;

  isNearest = (rounding_mode == rmNearestTiesToEven
               || rounding_mode == rmNearestTiesToAway);

  parts = partCountForBits(semantics->precision + 11);

  /* Calculate pow(5, abs(exp)).  */
  pow5PartCount = powerOf5(pow5Parts, exp >= 0 ? exp: -exp);

  for (;; parts *= 2) {
    opStatus sigStatus, powStatus;
    unsigned int excessPrecision, truncatedBits;

    calcSemantics.precision = parts * integerPartWidth - 1;
    excessPrecision = calcSemantics.precision - semantics->precision;
    truncatedBits = excessPrecision;

    APFloat decSig(calcSemantics, fcZero, sign);
    APFloat pow5(calcSemantics, fcZero, false);

    sigStatus = decSig.convertFromUnsignedParts(decSigParts, sigPartCount,
                                                rmNearestTiesToEven);
    powStatus = pow5.convertFromUnsignedParts(pow5Parts, pow5PartCount,
                                              rmNearestTiesToEven);
    /* Add exp, as 10^n = 5^n * 2^n.  */
    decSig.exponent += exp;

    lostFraction calcLostFraction;
    integerPart HUerr, HUdistance, powHUerr;

    if (exp >= 0) {
      /* multiplySignificand leaves the precision-th bit set to 1.  */
      calcLostFraction = decSig.multiplySignificand(pow5, NULL);
      powHUerr = powStatus != opOK;
    } else {
      calcLostFraction = decSig.divideSignificand(pow5);
      /* Denormal numbers have less precision.  */
      if (decSig.exponent < semantics->minExponent) {
        excessPrecision += (semantics->minExponent - decSig.exponent);
        truncatedBits = excessPrecision;
        if (excessPrecision > calcSemantics.precision)
          excessPrecision = calcSemantics.precision;
      }
      /* Extra half-ulp lost in reciprocal of exponent.  */
      powHUerr = (powStatus == opOK && calcLostFraction == lfExactlyZero) ? 0: 2;
    }

    /* Both multiplySignificand and divideSignificand return the
       result with the integer bit set.  */
    assert (APInt::tcExtractBit
            (decSig.significandParts(), calcSemantics.precision - 1) == 1);

    HUerr = HUerrBound(calcLostFraction != lfExactlyZero, sigStatus != opOK,
                       powHUerr);
    HUdistance = 2 * ulpsFromBoundary(decSig.significandParts(),
                                      excessPrecision, isNearest);

    /* Are we guaranteed to round correctly if we truncate?  */
    if (HUdistance >= HUerr) {
      APInt::tcExtract(significandParts(), partCount(), decSig.significandParts(),
                       calcSemantics.precision - excessPrecision,
                       excessPrecision);
      /* Take the exponent of decSig.  If we tcExtract-ed less bits
         above we must adjust our exponent to compensate for the
         implicit right shift.  */
      exponent = (decSig.exponent + semantics->precision
                  - (calcSemantics.precision - excessPrecision));
      calcLostFraction = lostFractionThroughTruncation(decSig.significandParts(),
                                                       decSig.partCount(),
                                                       truncatedBits);
      return normalize(rounding_mode, calcLostFraction);
    }
  }
}

APFloat::opStatus
APFloat::convertFromDecimalString(const char *p, roundingMode rounding_mode)
{
  decimalInfo D;
  opStatus fs;

  /* Scan the text.  */
  interpretDecimal(p, &D);

  /* Handle the quick cases.  First the case of no significant digits,
     i.e. zero, and then exponents that are obviously too large or too
     small.  Writing L for log 10 / log 2, a number d.ddddd*10^exp
     definitely overflows if

           (exp - 1) * L >= maxExponent

     and definitely underflows to zero where

           (exp + 1) * L <= minExponent - precision

     With integer arithmetic the tightest bounds for L are

           93/28 < L < 196/59            [ numerator <= 256 ]
           42039/12655 < L < 28738/8651  [ numerator <= 65536 ]
  */

  if (decDigitValue(*D.firstSigDigit) >= 10U) {
    category = fcZero;
    fs = opOK;
  } else if (D.res == resUnderflow ||
             (D.res == resOK && (D.normalizedExponent + 1) * 28738
              <= 8651 * (semantics->minExponent - (int) semantics->precision))) {
    /* Underflow to zero and round.  */
    zeroSignificand();
    fs = normalize(rounding_mode, lfLessThanHalf);
  } else if (D.res == resOverflow ||
             (D.res == resOK && (D.normalizedExponent - 1) * 42039
              >= 12655 * semantics->maxExponent)) {
    /* Overflow and round.  */
    fs = handleOverflow(rounding_mode);
  } else {
    integerPart *decSignificand;
    unsigned int partCount;

    /* A tight upper bound on number of bits required to hold an
       N-digit decimal integer is N * 196 / 59.  Allocate enough space
       to hold the full significand, and an extra part required by
       tcMultiplyPart.  */
    partCount = (D.lastSigDigit - D.firstSigDigit) + 1;
    partCount = partCountForBits(1 + 196 * partCount / 59);
    decSignificand = new integerPart[partCount + 1];
    partCount = 0;

    /* Convert to binary efficiently - we do almost all multiplication
       in an integerPart.  When this would overflow do we do a single
       bignum multiplication, and then revert again to multiplication
       in an integerPart.  */
    do {
      integerPart decValue, val, multiplier;

      val = 0;
      multiplier = 1;

      do {
        if (*p == '.')
          p++;

        decValue = decDigitValue(*p++);
        multiplier *= 10;
        val = val * 10 + decValue;
        /* The maximum number that can be multiplied by ten with any
           digit added without overflowing an integerPart.  */
      } while (p <= D.lastSigDigit && multiplier <= (~ (integerPart) 0 - 9) / 10);

      /* Multiply out the current part.  */
      APInt::tcMultiplyPart(decSignificand, decSignificand, multiplier, val,
                            partCount, partCount + 1, false);

      /* If we used another part (likely but not guaranteed), increase
         the count.  */
      if (decSignificand[partCount])
        partCount++;
    } while (p <= D.lastSigDigit);

    category = fcNormal;
    fs = roundSignificandWithExponent(decSignificand, partCount,
                                      D.exponent, rounding_mode);

    delete [] decSignificand;
  }

  return fs;
}

APFloat::opStatus
APFloat::convertFromString(const char *p, roundingMode rounding_mode)
{
  /* Handle a leading minus sign.  */
  if(*p == '-')
    sign = 1, p++;
  else
    sign = 0;

  if(p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
    return convertFromHexadecimalString(p + 2, rounding_mode);
  else
    return convertFromDecimalString(p, rounding_mode);
}

/* Write out a hexadecimal representation of the floating point value
   to DST, which must be of sufficient size, in the C99 form
   [-]0xh.hhhhp[+-]d.  Return the number of characters written,
   excluding the terminating NUL.

   If UPPERCASE, the output is in upper case, otherwise in lower case.

   HEXDIGITS digits appear altogether, rounding the value if
   necessary.  If HEXDIGITS is 0, the minimal precision to display the
   number precisely is used instead.  If nothing would appear after
   the decimal point it is suppressed.

   The decimal exponent is always printed and has at least one digit.
   Zero values display an exponent of zero.  Infinities and NaNs
   appear as "infinity" or "nan" respectively.

   The above rules are as specified by C99.  There is ambiguity about
   what the leading hexadecimal digit should be.  This implementation
   uses whatever is necessary so that the exponent is displayed as
   stored.  This implies the exponent will fall within the IEEE format
   range, and the leading hexadecimal digit will be 0 (for denormals),
   1 (normal numbers) or 2 (normal numbers rounded-away-from-zero with
   any other digits zero).
*/
unsigned int
APFloat::convertToHexString(char *dst, unsigned int hexDigits,
                            bool upperCase, roundingMode rounding_mode) const
{
  char *p;

  p = dst;
  if (sign)
    *dst++ = '-';

  switch (category) {
  case fcInfinity:
    memcpy (dst, upperCase ? infinityU: infinityL, sizeof infinityU - 1);
    dst += sizeof infinityL - 1;
    break;

  case fcNaN:
    memcpy (dst, upperCase ? NaNU: NaNL, sizeof NaNU - 1);
    dst += sizeof NaNU - 1;
    break;

  case fcZero:
    *dst++ = '0';
    *dst++ = upperCase ? 'X': 'x';
    *dst++ = '0';
    if (hexDigits > 1) {
      *dst++ = '.';
      memset (dst, '0', hexDigits - 1);
      dst += hexDigits - 1;
    }
    *dst++ = upperCase ? 'P': 'p';
    *dst++ = '0';
    break;

  case fcNormal:
    dst = convertNormalToHexString (dst, hexDigits, upperCase, rounding_mode);
    break;
  }

  *dst = 0;

  return dst - p;
}

/* Does the hard work of outputting the correctly rounded hexadecimal
   form of a normal floating point number with the specified number of
   hexadecimal digits.  If HEXDIGITS is zero the minimum number of
   digits necessary to print the value precisely is output.  */
char *
APFloat::convertNormalToHexString(char *dst, unsigned int hexDigits,
                                  bool upperCase,
                                  roundingMode rounding_mode) const
{
  unsigned int count, valueBits, shift, partsCount, outputDigits;
  const char *hexDigitChars;
  const integerPart *significand;
  char *p;
  bool roundUp;

  *dst++ = '0';
  *dst++ = upperCase ? 'X': 'x';

  roundUp = false;
  hexDigitChars = upperCase ? hexDigitsUpper: hexDigitsLower;

  significand = significandParts();
  partsCount = partCount();

  /* +3 because the first digit only uses the single integer bit, so
     we have 3 virtual zero most-significant-bits.  */
  valueBits = semantics->precision + 3;
  shift = integerPartWidth - valueBits % integerPartWidth;

  /* The natural number of digits required ignoring trailing
     insignificant zeroes.  */
  outputDigits = (valueBits - significandLSB () + 3) / 4;

  /* hexDigits of zero means use the required number for the
     precision.  Otherwise, see if we are truncating.  If we are,
     find out if we need to round away from zero.  */
  if (hexDigits) {
    if (hexDigits < outputDigits) {
      /* We are dropping non-zero bits, so need to check how to round.
         "bits" is the number of dropped bits.  */
      unsigned int bits;
      lostFraction fraction;

      bits = valueBits - hexDigits * 4;
      fraction = lostFractionThroughTruncation (significand, partsCount, bits);
      roundUp = roundAwayFromZero(rounding_mode, fraction, bits);
    }
    outputDigits = hexDigits;
  }

  /* Write the digits consecutively, and start writing in the location
     of the hexadecimal point.  We move the most significant digit
     left and add the hexadecimal point later.  */
  p = ++dst;

  count = (valueBits + integerPartWidth - 1) / integerPartWidth;

  while (outputDigits && count) {
    integerPart part;

    /* Put the most significant integerPartWidth bits in "part".  */
    if (--count == partsCount)
      part = 0;  /* An imaginary higher zero part.  */
    else
      part = significand[count] << shift;

    if (count && shift)
      part |= significand[count - 1] >> (integerPartWidth - shift);

    /* Convert as much of "part" to hexdigits as we can.  */
    unsigned int curDigits = integerPartWidth / 4;

    if (curDigits > outputDigits)
      curDigits = outputDigits;
    dst += partAsHex (dst, part, curDigits, hexDigitChars);
    outputDigits -= curDigits;
  }

  if (roundUp) {
    char *q = dst;

    /* Note that hexDigitChars has a trailing '0'.  */
    do {
      q--;
      *q = hexDigitChars[hexDigitValue (*q) + 1];
    } while (*q == '0');
    assert (q >= p);
  } else {
    /* Add trailing zeroes.  */
    memset (dst, '0', outputDigits);
    dst += outputDigits;
  }

  /* Move the most significant digit to before the point, and if there
     is something after the decimal point add it.  This must come
     after rounding above.  */
  p[-1] = p[0];
  if (dst -1 == p)
    dst--;
  else
    p[0] = '.';

  /* Finally output the exponent.  */
  *dst++ = upperCase ? 'P': 'p';

  return writeSignedDecimal (dst, exponent);
}

/* Normalized remainder.  This is not currently doing TRT.  */
APFloat::opStatus
APFloat::badmod(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;
  APFloat V = *this;
  unsigned int origSign = sign;

  fs = V.divide(rhs, rmNearestTiesToEven);
  if (fs == opDivByZero)
    return fs;

  int parts = partCount();
  integerPart *x = new integerPart[parts];

  fs = V.convertToInteger(x, parts * integerPartWidth, true,
                          rmNearestTiesToEven);
  if (fs==opInvalidOp)
    return fs;

  fs = V.convertFromSignExtendedInteger(x, parts * integerPartWidth, true,
                                        rmNearestTiesToEven);
  assert(fs==opOK);   // should always work

  fs = V.multiply(rhs, rounding_mode);
  assert(fs==opOK || fs==opInexact);   // should not overflow or underflow

  fs = subtract(V, rounding_mode);
  assert(fs==opOK || fs==opInexact);   // likewise

  if (isZero())
    sign = origSign;    // IEEE754 requires this
  delete[] x;
  return fs;
}
