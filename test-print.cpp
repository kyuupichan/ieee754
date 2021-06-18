/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include <cstdio>
#include "APFloat.h"

using namespace llvm;

#if 0
const fltSemantics *all_semantics[] = {
  &APFloat::IEEEsingle,
  &APFloat::IEEEdouble,
  &APFloat::IEEEquad,
  &APFloat::x87DoubleExtended,
};

static void
print (const char *a, const fltSemantics &kind,
       unsigned int digits, APFloat::roundingMode rounding_mode)
{
  char str[128];
  APFloat f (kind, a);

  f.convertToHexString(str, digits, false, rounding_mode);
  printf ("%s\n", str);
}

static void
print (const fltSemantics &kind, bool sign, APFloat::fltCategory category,
       APFloat::roundingMode rounding_mode)
{
  char str[128];
  APFloat f (kind, category, sign);

  f.convertToHexString(str, 0, false, rounding_mode);
  printf ("%s\n", str);
}

int main (void)
{
  char p[100];
  unsigned int digits;
  const char **str, *testsF[] = {
    "0x1.718764p-126",
    "-0x1.718p53",
    "0x1.708p5",
    "0x1.ffff8p0",
    "0x0p80",
    0
  };

  const char *testsD[] = {
    "-0x1.456789abcdef00p524",
    "0x1p-1074",
    0
  };

  const char *testsQ[] = {
    "0x1.fedcba9876543210fedcba987p3256",
    "0x1.ffp0",
    0
  };

  /*  rmNearestTiesToEven,
      rmTowardPositive,
      rmTowardNegative,
      rmTowardZero,
      rmNearestTiesToAway  */

  for (str = testsF; *str; str++) {
    for (int i = 0; i < 5; i++) {
      for (digits = 0; digits < 9; digits++) {
	APFloat::roundingMode rm ((APFloat::roundingMode) i);
	print (*str, APFloat::IEEEsingle, digits, rm);
      }
    }
  }

  for (str = testsD; *str; str++) {
    for (int i = 0; i < 5; i++) {
      for (digits = 0; digits < 16; digits++) {
	APFloat::roundingMode rm ((APFloat::roundingMode) i);
	print (*str, APFloat::IEEEdouble, digits, rm);
      }
    }
  }

  for (str = testsQ; *str; str++) {
    for (int i = 0; i < 5; i++) {
      for (digits = 0; digits < 28; digits++) {
	APFloat::roundingMode rm ((APFloat::roundingMode) i);
	print (*str, APFloat::IEEEquad, digits, rm);
      }
    }
  }

  for (int j = 0; j < 4; j++)
    {
      const fltSemantics &kind = *all_semantics[j];

      print (kind, false, APFloat::fcQNaN, APFloat::rmTowardZero);
      print (kind, true, APFloat::fcQNaN, APFloat::rmTowardZero);
      print (kind, false, APFloat::fcInfinity, APFloat::rmTowardZero);
      print (kind, true, APFloat::fcInfinity, APFloat::rmTowardZero);
    }

  return 0;
}

/* A tight upper bound on number of parts required to hold the value
   pow (5, power) is

     power * 1024 / (441 * integerPartWidth) + 1

   However, whilst the result may require only this many parts,
   because we are multiplying two values to get it, the multiplication
   may require an extra part with the excess part being zero (consider
   the trivial case of 1 * 1, tcFullMultiply requires two parts to
   hold the single-part result).  So we add an extra one to guarantee
   enough space whilst multiplying.  */
const unsigned int maxExponent = 16383;
const unsigned int maxParts = 2 + ((maxExponent * 1024)
				   / (441 * integerPartWidth));

COMPILE_TIME_ASSERT(integerPartWidth >= 19 );	/* Could be relaxed.  */
static unsigned int
powerOf5 (integerPart *dst, unsigned int power)
{
  static integerPart firstEightPowers[] = { 1, 5, 25, 125, 625, 3125,
					    15625, 78125 };
  static integerPart pow5s[maxParts * 2 + 5] = { 78125 * 5 };
  static unsigned int partsCount[16] = { 1 };

  integerPart scratch[maxParts], *p1, *p2, *pow5;
  unsigned int result;

  assert (power <= maxExponent);

  p1 = dst;
  p2 = scratch;

  *p1 = firstEightPowers[power & 7];
  power >>= 3;

  result = 1;
  pow5 = pow5s;

  for (unsigned int n = 0; power; power >>= 1, n++) {
    unsigned int pc;

    pc = partsCount[n];

    /* Calculate pow(5, pow(2, n + 3)) if we haven't yet.  */
    if (pc == 0) {
      pc = partsCount[n - 1];
      pc = APInt::tcFullMultiply (pow5, pow5 - pc, pow5 - pc, pc, pc);
      partsCount[n] = pc;
    }

    if (power & 1) {
      integerPart *tmp;

      result = APInt::tcFullMultiply (p2, p1, pow5, result, pc);

      /* Now result is in p1 with partsCount parts and p2 is scratch
	 space.  */
      tmp = p1, p1 = p2, p2 = tmp;
    }

    pow5 += pc;
  }

  if (p1 != dst)
    APInt::tcAssign (dst, p1, result);

  return result;
}
#endif

int main (void)
{
  APFloat epsilon(APFloat::IEEEquad, "0x1p-53");
  APFloat one(APFloat::IEEEquad, "1.0");
  APFloat oned(APFloat::IEEEdouble, "1.0");

  char buf[100];

  APFloat lhs(APFloat::IEEEdouble, "2251799813685248.5");
  //APFloat rhs(APFloat::IEEEdouble, "0x80000000000004000000.010p-28");
  APFloat rhs(APFloat::IEEEdouble, "0x80000000000004000000.010p-28");

  lhs.convertToHexString(buf, 0, false, APFloat::rmNearestTiesToEven);
  puts(buf);
  rhs.convertToHexString(buf, 0, false, APFloat::rmNearestTiesToEven);
  puts(buf);

  // APFloat x (APFloat::IEEEsingle, "0x1.345672p+30");
  // APFloat y (APFloat::IEEEsingle, "0x1.56789ap-20");

  // x.convertToHexString (dst, 0, false, APFloat::rmNearestTiesToEven);
  // printf ("x: %s\n", dst);
  // y.convertToHexString (dst, 0, false, APFloat::rmNearestTiesToEven);
  // printf ("y: %s\n", dst);
  // x.badmod(y, APFloat::rmNearestTiesToEven);
  // x.convertToHexString (dst, 0, false, APFloat::rmNearestTiesToEven);
  // printf ("mod: %s\n", dst);

  return 0;
}
