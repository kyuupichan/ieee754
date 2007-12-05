/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include "APFloat.h"

using namespace llvm;

int main (void)
{
  APFloat::opStatus status;

  // Initialize it.  C++ constructors can't return values :(
  APFloat d (APFloat::IEEEdouble, APFloat::fcZero, false);

  // Simple cases.
  status = d.convertFromString("1.5", APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);

  status = d.convertFromString("1.4", APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opInexact);

  status = d.convertFromString("0x1.4p2", APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);

  status = d.convertFromString("0.0", APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);
  assert (d.getCategory() == APFloat::fcZero);

  status = d.convertFromString("0.0f", APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);
  assert (d.getCategory() == APFloat::fcZero);

  // Just fits
  status = d.convertFromString("0x1.4444444444444p2",
                               APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opOK);

  // A bit too wide
  status = d.convertFromString("0x1.44444444444448p2",
                               APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opInexact);

  // As I mentioned in the email thread, for some very long decimals,
  // a speed optimization in the decimal->binary conversion causes the
  // function to occasionally report inexact even though the number
  // can be represented exactly.  The resulting APFloat is always
  // correct; just the return status is inaccurate.  It is fairly
  // rare, fixing it is a not-too-hard exercise.  This number is 1 +
  // 2^-40 which can be represented exactly.
  status = d.convertFromString("1.0000000000009094947017729282379150390625",
                               APFloat::rmNearestTiesToEven);
  assert (status == APFloat::opInexact); // Should be opOK.

  return 0;
}
