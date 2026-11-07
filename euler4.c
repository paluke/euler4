/*
 * euler4.c -- search for primitive integer solutions of
 *
 *      a^4 + b^4 + c^4 = d^4
 *
 * Build:  gcc -O3 -fopenmp -march=native -DNDEBUG euler4.c
 * Run:    ./search [end]  or  ./search <start> <end>
 *         (default end = 3000000, start = 1)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _OPENMP
#define USE_COUNTERS
#endif

typedef unsigned __int128 u128;

#define CNT(x) (sizeof(x)/sizeof(*x)/!(sizeof(x)%sizeof(*x)))

#define INV8B(x) (x*3ull^2)*(2-x*(x*3ull^2))
#define INV16B(x) INV8B(x)*(2-x*INV8B(x))
#define INV32B(x) INV16B(x)*(2-x*INV16B(x))
#define INV64B(x) INV32B(x)*(2-x*INV32B(x))

static const unsigned small_prime[] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 
    37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 
    181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 
    293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 
    359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 
    479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 
    613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 
    673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 
    743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 
    821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 
    881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 
    953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 
    1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 
    1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 
    1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 
    1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 
    1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 
    1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 
    1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 
    1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 
    1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 
    1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 
    1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 
    1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 
    1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 
    1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039
};

static const uint64_t small_prime_inv[] = {
    INV64B(3), INV64B(5), INV64B(7), INV64B(11), INV64B(13), INV64B(17), INV64B(19), INV64B(23), INV64B(29), INV64B(31), 
    INV64B(37), INV64B(41), INV64B(43), INV64B(47), INV64B(53), INV64B(59), INV64B(61), INV64B(67), INV64B(71), INV64B(73), 
    INV64B(79), INV64B(83), INV64B(89), INV64B(97), INV64B(101), INV64B(103), INV64B(107), INV64B(109), INV64B(113), INV64B(127), 
    INV64B(131), INV64B(137), INV64B(139), INV64B(149), INV64B(151), INV64B(157), INV64B(163), INV64B(167), INV64B(173), INV64B(179), 
    INV64B(181), INV64B(191), INV64B(193), INV64B(197), INV64B(199), INV64B(211), INV64B(223), INV64B(227), INV64B(229), INV64B(233), 
    INV64B(239), INV64B(241), INV64B(251), INV64B(257), INV64B(263), INV64B(269), INV64B(271), INV64B(277), INV64B(281), INV64B(283), 
    INV64B(293), INV64B(307), INV64B(311), INV64B(313), INV64B(317), INV64B(331), INV64B(337), INV64B(347), INV64B(349), INV64B(353), 
    INV64B(359), INV64B(367), INV64B(373), INV64B(379), INV64B(383), INV64B(389), INV64B(397), INV64B(401), INV64B(409), INV64B(419), 
    INV64B(421), INV64B(431), INV64B(433), INV64B(439), INV64B(443), INV64B(449), INV64B(457), INV64B(461), INV64B(463), INV64B(467), 
    INV64B(479), INV64B(487), INV64B(491), INV64B(499), INV64B(503), INV64B(509), INV64B(521), INV64B(523), INV64B(541), INV64B(547), 
    INV64B(557), INV64B(563), INV64B(569), INV64B(571), INV64B(577), INV64B(587), INV64B(593), INV64B(599), INV64B(601), INV64B(607), 
    INV64B(613), INV64B(617), INV64B(619), INV64B(631), INV64B(641), INV64B(643), INV64B(647), INV64B(653), INV64B(659), INV64B(661), 
    INV64B(673), INV64B(677), INV64B(683), INV64B(691), INV64B(701), INV64B(709), INV64B(719), INV64B(727), INV64B(733), INV64B(739), 
    INV64B(743), INV64B(751), INV64B(757), INV64B(761), INV64B(769), INV64B(773), INV64B(787), INV64B(797), INV64B(809), INV64B(811), 
    INV64B(821), INV64B(823), INV64B(827), INV64B(829), INV64B(839), INV64B(853), INV64B(857), INV64B(859), INV64B(863), INV64B(877), 
    INV64B(881), INV64B(883), INV64B(887), INV64B(907), INV64B(911), INV64B(919), INV64B(929), INV64B(937), INV64B(941), INV64B(947), 
    INV64B(953), INV64B(967), INV64B(971), INV64B(977), INV64B(983), INV64B(991), INV64B(997), INV64B(1009), INV64B(1013), INV64B(1019), 
    INV64B(1021), INV64B(1031), INV64B(1033), INV64B(1039), INV64B(1049), INV64B(1051), INV64B(1061), INV64B(1063), INV64B(1069), INV64B(1087), 
    INV64B(1091), INV64B(1093), INV64B(1097), INV64B(1103), INV64B(1109), INV64B(1117), INV64B(1123), INV64B(1129), INV64B(1151), INV64B(1153), 
    INV64B(1163), INV64B(1171), INV64B(1181), INV64B(1187), INV64B(1193), INV64B(1201), INV64B(1213), INV64B(1217), INV64B(1223), INV64B(1229), 
    INV64B(1231), INV64B(1237), INV64B(1249), INV64B(1259), INV64B(1277), INV64B(1279), INV64B(1283), INV64B(1289), INV64B(1291), INV64B(1297), 
    INV64B(1301), INV64B(1303), INV64B(1307), INV64B(1319), INV64B(1321), INV64B(1327), INV64B(1361), INV64B(1367), INV64B(1373), INV64B(1381), 
    INV64B(1399), INV64B(1409), INV64B(1423), INV64B(1427), INV64B(1429), INV64B(1433), INV64B(1439), INV64B(1447), INV64B(1451), INV64B(1453), 
    INV64B(1459), INV64B(1471), INV64B(1481), INV64B(1483), INV64B(1487), INV64B(1489), INV64B(1493), INV64B(1499), INV64B(1511), INV64B(1523), 
    INV64B(1531), INV64B(1543), INV64B(1549), INV64B(1553), INV64B(1559), INV64B(1567), INV64B(1571), INV64B(1579), INV64B(1583), INV64B(1597), 
    INV64B(1601), INV64B(1607), INV64B(1609), INV64B(1613), INV64B(1619), INV64B(1621), INV64B(1627), INV64B(1637), INV64B(1657), INV64B(1663), 
    INV64B(1667), INV64B(1669), INV64B(1693), INV64B(1697), INV64B(1699), INV64B(1709), INV64B(1721), INV64B(1723), INV64B(1733), INV64B(1741), 
    INV64B(1747), INV64B(1753), INV64B(1759), INV64B(1777), INV64B(1783), INV64B(1787), INV64B(1789), INV64B(1801), INV64B(1811), INV64B(1823), 
    INV64B(1831), INV64B(1847), INV64B(1861), INV64B(1867), INV64B(1871), INV64B(1873), INV64B(1877), INV64B(1879), INV64B(1889), INV64B(1901), 
    INV64B(1907), INV64B(1913), INV64B(1931), INV64B(1933), INV64B(1949), INV64B(1951), INV64B(1973), INV64B(1979), INV64B(1987), INV64B(1993), 
    INV64B(1997), INV64B(1999), INV64B(2003), INV64B(2011), INV64B(2017), INV64B(2027), INV64B(2029), INV64B(2039)
};

static const uint64_t small_prime_lim[] = {
    UINT64_MAX/3, UINT64_MAX/5, UINT64_MAX/7, UINT64_MAX/11, UINT64_MAX/13, UINT64_MAX/17, UINT64_MAX/19, UINT64_MAX/23, UINT64_MAX/29, UINT64_MAX/31, 
    UINT64_MAX/37, UINT64_MAX/41, UINT64_MAX/43, UINT64_MAX/47, UINT64_MAX/53, UINT64_MAX/59, UINT64_MAX/61, UINT64_MAX/67, UINT64_MAX/71, UINT64_MAX/73, 
    UINT64_MAX/79, UINT64_MAX/83, UINT64_MAX/89, UINT64_MAX/97, UINT64_MAX/101, UINT64_MAX/103, UINT64_MAX/107, UINT64_MAX/109, UINT64_MAX/113, UINT64_MAX/127, 
    UINT64_MAX/131, UINT64_MAX/137, UINT64_MAX/139, UINT64_MAX/149, UINT64_MAX/151, UINT64_MAX/157, UINT64_MAX/163, UINT64_MAX/167, UINT64_MAX/173, UINT64_MAX/179, 
    UINT64_MAX/181, UINT64_MAX/191, UINT64_MAX/193, UINT64_MAX/197, UINT64_MAX/199, UINT64_MAX/211, UINT64_MAX/223, UINT64_MAX/227, UINT64_MAX/229, UINT64_MAX/233, 
    UINT64_MAX/239, UINT64_MAX/241, UINT64_MAX/251, UINT64_MAX/257, UINT64_MAX/263, UINT64_MAX/269, UINT64_MAX/271, UINT64_MAX/277, UINT64_MAX/281, UINT64_MAX/283, 
    UINT64_MAX/293, UINT64_MAX/307, UINT64_MAX/311, UINT64_MAX/313, UINT64_MAX/317, UINT64_MAX/331, UINT64_MAX/337, UINT64_MAX/347, UINT64_MAX/349, UINT64_MAX/353, 
    UINT64_MAX/359, UINT64_MAX/367, UINT64_MAX/373, UINT64_MAX/379, UINT64_MAX/383, UINT64_MAX/389, UINT64_MAX/397, UINT64_MAX/401, UINT64_MAX/409, UINT64_MAX/419, 
    UINT64_MAX/421, UINT64_MAX/431, UINT64_MAX/433, UINT64_MAX/439, UINT64_MAX/443, UINT64_MAX/449, UINT64_MAX/457, UINT64_MAX/461, UINT64_MAX/463, UINT64_MAX/467, 
    UINT64_MAX/479, UINT64_MAX/487, UINT64_MAX/491, UINT64_MAX/499, UINT64_MAX/503, UINT64_MAX/509, UINT64_MAX/521, UINT64_MAX/523, UINT64_MAX/541, UINT64_MAX/547, 
    UINT64_MAX/557, UINT64_MAX/563, UINT64_MAX/569, UINT64_MAX/571, UINT64_MAX/577, UINT64_MAX/587, UINT64_MAX/593, UINT64_MAX/599, UINT64_MAX/601, UINT64_MAX/607, 
    UINT64_MAX/613, UINT64_MAX/617, UINT64_MAX/619, UINT64_MAX/631, UINT64_MAX/641, UINT64_MAX/643, UINT64_MAX/647, UINT64_MAX/653, UINT64_MAX/659, UINT64_MAX/661, 
    UINT64_MAX/673, UINT64_MAX/677, UINT64_MAX/683, UINT64_MAX/691, UINT64_MAX/701, UINT64_MAX/709, UINT64_MAX/719, UINT64_MAX/727, UINT64_MAX/733, UINT64_MAX/739, 
    UINT64_MAX/743, UINT64_MAX/751, UINT64_MAX/757, UINT64_MAX/761, UINT64_MAX/769, UINT64_MAX/773, UINT64_MAX/787, UINT64_MAX/797, UINT64_MAX/809, UINT64_MAX/811, 
    UINT64_MAX/821, UINT64_MAX/823, UINT64_MAX/827, UINT64_MAX/829, UINT64_MAX/839, UINT64_MAX/853, UINT64_MAX/857, UINT64_MAX/859, UINT64_MAX/863, UINT64_MAX/877, 
    UINT64_MAX/881, UINT64_MAX/883, UINT64_MAX/887, UINT64_MAX/907, UINT64_MAX/911, UINT64_MAX/919, UINT64_MAX/929, UINT64_MAX/937, UINT64_MAX/941, UINT64_MAX/947, 
    UINT64_MAX/953, UINT64_MAX/967, UINT64_MAX/971, UINT64_MAX/977, UINT64_MAX/983, UINT64_MAX/991, UINT64_MAX/997, UINT64_MAX/1009, UINT64_MAX/1013, UINT64_MAX/1019, 
    UINT64_MAX/1021, UINT64_MAX/1031, UINT64_MAX/1033, UINT64_MAX/1039, UINT64_MAX/1049, UINT64_MAX/1051, UINT64_MAX/1061, UINT64_MAX/1063, UINT64_MAX/1069, UINT64_MAX/1087, 
    UINT64_MAX/1091, UINT64_MAX/1093, UINT64_MAX/1097, UINT64_MAX/1103, UINT64_MAX/1109, UINT64_MAX/1117, UINT64_MAX/1123, UINT64_MAX/1129, UINT64_MAX/1151, UINT64_MAX/1153, 
    UINT64_MAX/1163, UINT64_MAX/1171, UINT64_MAX/1181, UINT64_MAX/1187, UINT64_MAX/1193, UINT64_MAX/1201, UINT64_MAX/1213, UINT64_MAX/1217, UINT64_MAX/1223, UINT64_MAX/1229, 
    UINT64_MAX/1231, UINT64_MAX/1237, UINT64_MAX/1249, UINT64_MAX/1259, UINT64_MAX/1277, UINT64_MAX/1279, UINT64_MAX/1283, UINT64_MAX/1289, UINT64_MAX/1291, UINT64_MAX/1297, 
    UINT64_MAX/1301, UINT64_MAX/1303, UINT64_MAX/1307, UINT64_MAX/1319, UINT64_MAX/1321, UINT64_MAX/1327, UINT64_MAX/1361, UINT64_MAX/1367, UINT64_MAX/1373, UINT64_MAX/1381, 
    UINT64_MAX/1399, UINT64_MAX/1409, UINT64_MAX/1423, UINT64_MAX/1427, UINT64_MAX/1429, UINT64_MAX/1433, UINT64_MAX/1439, UINT64_MAX/1447, UINT64_MAX/1451, UINT64_MAX/1453, 
    UINT64_MAX/1459, UINT64_MAX/1471, UINT64_MAX/1481, UINT64_MAX/1483, UINT64_MAX/1487, UINT64_MAX/1489, UINT64_MAX/1493, UINT64_MAX/1499, UINT64_MAX/1511, UINT64_MAX/1523, 
    UINT64_MAX/1531, UINT64_MAX/1543, UINT64_MAX/1549, UINT64_MAX/1553, UINT64_MAX/1559, UINT64_MAX/1567, UINT64_MAX/1571, UINT64_MAX/1579, UINT64_MAX/1583, UINT64_MAX/1597, 
    UINT64_MAX/1601, UINT64_MAX/1607, UINT64_MAX/1609, UINT64_MAX/1613, UINT64_MAX/1619, UINT64_MAX/1621, UINT64_MAX/1627, UINT64_MAX/1637, UINT64_MAX/1657, UINT64_MAX/1663, 
    UINT64_MAX/1667, UINT64_MAX/1669, UINT64_MAX/1693, UINT64_MAX/1697, UINT64_MAX/1699, UINT64_MAX/1709, UINT64_MAX/1721, UINT64_MAX/1723, UINT64_MAX/1733, UINT64_MAX/1741, 
    UINT64_MAX/1747, UINT64_MAX/1753, UINT64_MAX/1759, UINT64_MAX/1777, UINT64_MAX/1783, UINT64_MAX/1787, UINT64_MAX/1789, UINT64_MAX/1801, UINT64_MAX/1811, UINT64_MAX/1823, 
    UINT64_MAX/1831, UINT64_MAX/1847, UINT64_MAX/1861, UINT64_MAX/1867, UINT64_MAX/1871, UINT64_MAX/1873, UINT64_MAX/1877, UINT64_MAX/1879, UINT64_MAX/1889, UINT64_MAX/1901, 
    UINT64_MAX/1907, UINT64_MAX/1913, UINT64_MAX/1931, UINT64_MAX/1933, UINT64_MAX/1949, UINT64_MAX/1951, UINT64_MAX/1973, UINT64_MAX/1979, UINT64_MAX/1987, UINT64_MAX/1993, 
    UINT64_MAX/1997, UINT64_MAX/1999, UINT64_MAX/2003, UINT64_MAX/2011, UINT64_MAX/2017, UINT64_MAX/2027, UINT64_MAX/2029, UINT64_MAX/2039
};



#define PERIOD  (625ULL * 16384ULL)

/* d*d and (d^2+c^2)/2 must fit in uint64_t.  Since d = 1 (mod 8),
   the largest value actually checked with this limit is 2^32-7. */
#define MAX_D_LIMIT ((1ULL << 32) - 1)

/* ------------------------------------------------------------------ */
/* 64-bit factorisation                                               */
/* ------------------------------------------------------------------ */

/* Any 64-bit number has at most 15 distinct prime factors because
   2*3*5*...*53 > 2^64; 16 slots are enough. */
#define MAX_FACTORS 16

typedef struct {
    uint64_t p;
    unsigned e;
} factor_t;

/* ------------------------------------------------------------------ */
/* smallest-prime-factor table for numbers coprime to 30              */
/* ------------------------------------------------------------------ */

#define FACTOR_TABLE_LIMIT 20000000u

static uint16_t *spf_table;

#ifdef USE_COUNTERS
uint64_t    pollard_counter = 0;
#endif

/* A part of the factorization still to be processed: v is its value and
   mult is the multiplicity of this part in the original n. */
typedef struct {
    uint64_t v;
    unsigned mult;
} pending_t;

/* Residues for c modulo PERIOD, filled by make_residue_table() before
   the parallel loop. */
static uint32_t residue_table[774];

/* Solution log; report_solution() updates it under the critical section. */
#define MAX_SOL 1024
static uint64_t sol_a[MAX_SOL], sol_b[MAX_SOL], sol_c[MAX_SOL], sol_d[MAX_SOL];
static int nsol;

/* ------------------------------------------------------------------ */
/* data used by the Gaussian reconstruction                           */
/* ------------------------------------------------------------------ */

typedef struct { int64_t re, im; } gauss_t;

/* Read-only table of representations for small good primes.  It is
   built before the parallel loop and then only read by the workers. */
#define GAUSS_PRECOMP_LIMIT 10000000u

#if GAUSS_PRECOMP_LIMIT > FACTOR_TABLE_LIMIT
#error "GAUSS_PRECOMP_LIMIT must not exceed FACTOR_TABLE_LIMIT"
#endif

typedef struct {
    uint32_t p, A, B;
} gauss_pre_t;

static gauss_pre_t *gauss_pre;
static size_t gauss_pre_mask;

typedef struct {
    factor_t *good;
    int ng;
    int delta2;
    u128 R;
    uint64_t q, c, d;
    int found;
} enum_ctx;

/* ------------------------------------------------------------------ */
/* small arithmetic helpers                                           */
/* ------------------------------------------------------------------ */

/* a,b < m; return a * b % m */
static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m)
{
  assert (a < m);
  assert (b < m);
#ifndef __x86_64__
  return (unsigned __int128)a * b % m;
#else
  uint64_t quot, rem;
  asm("mulq %3\n\tdivq %4" : "=a"(quot), "=&d"(rem) : "%a"(a), "r"(b), "r"(m) : "cc");
  return rem;
#endif
}

/* a,b < m; return (a + b) % m; no overflow when a + b > 2**64 */
static inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m)
{
  assert (a < m);
  assert (b < m);
  return (b >= m - a) ? b - (m - a) : b + a;
}

/* a < m; return 2 * a % m; no overflow when 2 * a > 2**64 */
static inline uint64_t mul2mod(uint64_t a, uint64_t m)
{
  assert (a < m);
  return (a >= m - a) ? a - (m - a) : a + a;
}

/* a,b < m; return (a - b) % m */
static inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m)
{
  assert (a < m);
  assert (b < m);
  return (a >= b) ? a - b : m + a - b;
}

/* return a % b */
static inline uint64_t reminder(unsigned __int128 a, uint64_t b)
{
#ifndef __x86_64__
  return a % b;
#else
  uint64_t result;
  asm("divq %3" : "=d"(result): "a"((uint64_t)a), "d"((uint64_t)(a >> 64) % b), "r"(b) : "cc");
  return result;
#endif
}

/* inverse of an odd n modulo 2^64 */
static inline uint64_t mod_inv_2_64(uint64_t n) {
  assert(n % 2 == 1);
  uint64_t x = (3*n)^2; 
  uint64_t y  = 1 - n*x;
  x = x*(1 + y); y *= y;
  x = x*(1 + y); y *= y;
  x = x*(1 + y); y *= y;
  return x*(1 + y);
}

static uint64_t binary_gcd_odd(uint64_t  a, uint64_t  b)
{
  /* both must be odd */
  assert (a & b & 1);

  /* after right shift high bit will be zero */
  a >>= 1;
  b >>= 1;

  uint64_t c;
  while (0 != (c = a - b))
  {
      /* all mask bits will be 1 or 0 depend on sign bit of c */
      uint64_t mask = (uint64_t)((int64_t)c >> 63);

      /* calc min and abs without conditional branching */
      a = (c ^ mask) - mask; /* abs(a - b) */
      b += (mask & c); /* min(a, b) */

      a = (a >> 1) >> __builtin_ctzll(c);
  }
  return a + a + 1;
}

/* return gcd; first arg must be odd */
static inline uint64_t binary_gcd_odd1(uint64_t a, uint64_t b)
{
  unsigned k = __builtin_ctzll(b);
  return binary_gcd_odd(a, b >> k);
}

static uint32_t is_square_odd(uint64_t nn) {
  static const unsigned QUADRATIC_RESIDUES_MOD_35 = ((1 << 0) | (1 << 1) | (1 << 4) | (1 << 9) | (1 << 11) | (1 << 14) | (1 << 15) | (1 << 16) | (1 << 21) | (1 << 25) | (1 << 29) | (1 << 30));

  static const unsigned char inv0[128] = {
    1, 171, 167, 205, 143, 5, 73, 253, 31, 75, 57, 45, 175, 101, 215, 93, 63,
    21, 231, 115, 207, 197, 9, 67, 95, 117, 7, 237, 239, 219, 233, 227, 127,
    213, 217, 77, 241, 123, 55, 125, 159, 203, 71, 83, 209, 27, 169, 35, 191,
    107, 153, 243, 177, 69, 119, 195, 223, 11, 135, 109, 145, 165, 105, 157,
    255, 85, 89, 51, 113, 251, 183, 3, 225, 181, 199, 211, 81, 155, 41, 163,
    193, 235, 25, 141, 49, 59, 247, 189, 161, 139, 249, 19, 17, 37, 23, 29,
    129, 43, 39, 179, 15, 133, 201, 131, 97, 53, 185, 173, 47, 229, 87, 221,
    65, 149, 103, 13, 79, 187, 137, 61, 33, 245, 121, 147, 111, 91, 151, 99
  };

  if (nn % 8 != 1)
    return 0;

  /* if ((nn + 1) * 0xAAAAAAAAAAAAAAABull <= 0x5555555555555555ull) */
  if ((nn + 1) * (0 - ~0ull / 3) <= ~0ull / 3)
    return 0; /* n % 3 == 2 */

  if (((1ull << nn % 35) & QUADRATIC_RESIDUES_MOD_35) == 0)
    return 0;

  uint32_t inv = inv0[nn / 8 & 127]; // ((nn & 1024) ? 254 : 0) ^ inv0[nn / 8 & 63]

  inv *= (3 - inv * inv * (uint32_t)nn) / 2;

  uint64_t r = nn * inv;
  r *= (3 - inv * r) / 2;

  if (r & (1ull << 32))
    r = (uint32_t)(0 - r);
  else
    r = (uint32_t)r;

  if (r * r == nn)
    return r;

  return 0;
}

/* check for square; return sqrt if square else 0 */
static inline uint32_t is_square(uint64_t nn)
{
  if (nn == 0)
    return 0;
  unsigned k = __builtin_ctzll(nn) / 2;
  return is_square_odd(nn >> 2 * k) << k;
}

static uint64_t powmod(uint64_t base, uint64_t power, uint64_t modul)
{
  uint64_t result = 1;
  base = base % modul;
  for(;;)
  {
    if (power & 1)
      result = mulmod(result, base, modul);
    if (0 == (power >>= 1)) break;
    base = mulmod(base, base, modul);
  }
  return result;
}

static int jacobi(uint64_t a, uint64_t n)
{
    assert(n % 2 == 1);
    assert(a < n);
    int t = 2;
    uint64_t r;

    while (a > 1)
    {
        while (a % 4 == 0)
            a >>= 2;
        if (a % 2 == 0)
        {
            a >>= 1;
            r = n % 8;
            if (r == 3 || r == 5)
                t ^= 2;
        }

        r = n;
        n = a;
        a = r;
        if ((a & 3 & n) == 3)
            t ^= 2;

        a = a % n;
    }

    if (a == 1 || n == 1)
        return t - 1;
    else
        return 0;
}

static __attribute__((noinline)) unsigned min_nonres(uint64_t n)
{
  assert(n % 2 == 1);

  /* check for n % 8 in 3, 5 */
  if ((n ^ (n >> 1)) & 2)
    return 2;

  uint32_t inv = (n >> 1) & 1;

  if (n % 3 == 2 - inv)
    return 3;
  if (n % 5 == 2 || n % 5 == 3)
    return 5;

  unsigned i;

  /* 7 .. 61 */
  static uint64_t mask64[] = {
     0x17,
     0x23b,
     0x161b,
     0x1a317,
     0x30af3,
     0x5335f,
     0x13d122f3,
     0x121d47b7,
     0x165e211e9bLL,
     0x1b382b50737LL,
     0x35883a3ee53LL,
     0x4351b2753dfLL,
     0x12dd703303aed3LL,
     0x22b62183e7b92bbLL,
     0x1713e6940a59f23bLL
  };

  for (i = 0; i < CNT(mask64); ++i)
  {
     unsigned j = i + 2; /* skip 3, 5 */
     uint8_t mod = (uint8_t)(n % small_prime[j]);
     if (((small_prime[j] >> 1) & inv) != !((1LL << mod) & mask64[i]))
        return small_prime[j];
  }

  /* 67 .. 127 */
  static unsigned __int128 mask128[] = {
     ((unsigned __int128)0x3ULL<<64)|0x59c281ba27ebc653ULL,
     ((unsigned __int128)0x1ULL<<64)|0x164729716b1d977fULL,
     ((unsigned __int128)0x1ebULL<<64)|0x22c742790b8d135fULL,
     ((unsigned __int128)0x130bULL<<64)|0x409e755186fd2f37ULL,
     ((unsigned __int128)0x26873ULL<<64)|0xa80b1372fea31e9bULL,
     ((unsigned __int128)0x1b3c3b9ULL<<64)|0x2a6b59502770f37ULL,
     ((unsigned __int128)0x1eb628b47ULL<<64)|0x6067981b8b451b5fULL,
     ((unsigned __int128)0x1391b7f0d3ULL<<64)|0x552a832c3fb6273ULL,
     ((unsigned __int128)0x16380e9115ULL<<64)|0xbd964257768fe397ULL,
     ((unsigned __int128)0x27816ea9821ULL<<64)|0x633397be6a897e1bULL,
     ((unsigned __int128)0x1752639f4e85ULL<<64)|0xb003685cbe7192bbULL,
     ((unsigned __int128)0x1a75e89ae2121ULL<<64)|0xf33e1211d645eb97ULL,
     ((unsigned __int128)0x172a099c419697f1ULL<<64)|0x7016967dc66fab17ULL
  };

  for (i = 0; i < CNT(mask128); ++i)
  {
     unsigned j = i + 2 + CNT(mask64);
     uint8_t mod = (uint8_t)(n % small_prime[j]);
     if (((small_prime[j] >> 1) & inv) != !(((unsigned __int128)1 << mod) & mask128[i]))
        return small_prime[j];
  }

  if (is_square_odd(n) != 0)
    return 0;

  uint64_t k;
  for (k = 131; ; k += 6)
  {
     if(jacobi(k, n) == -1)
       return k;
     if(jacobi(k + 2, n) == -1)
       return k + 2;
  }
}

/* ------------------------------------------------------------------ */
/* Montgomery arithmetic for odd n < 2^64                              */
/*                                                                     */
/* Subtraction form of REDC: both the high part of a*b and the high    */
/* part of m*n are computed separately, so no 128-bit a*b + m*n sum    */
/* is needed.  This works even when n has the high bit set.  nr must   */
/* be the positive inverse n^{-1} mod 2^64.                            */
/* ------------------------------------------------------------------ */

typedef struct {
  uint64_t n;
  uint64_t nr;    /* n^{-1} mod 2^64 */
//  uint64_t r2;    /* 2^128 mod n */
} mont_t;

static inline uint64_t mont_mul(const mont_t *mo, uint64_t a, uint64_t b)
{
  unsigned __int128 t = (unsigned __int128)a * b;
  uint64_t u = (uint64_t)(t >> 64);
  uint64_t m = (uint64_t)t * mo->nr;
  m = (uint64_t)((unsigned __int128)m * mo->n >> 64);
  return u >= m ? u - m : u - m + mo->n;
}

static inline uint64_t mont_add(uint64_t a, uint64_t b, uint64_t n)
{
  uint64_t s = a + b;
  return (s < a || s >= n) ? s - n : s;
}

static inline uint64_t gcd_odd(uint64_t a, uint64_t b)
{
  if (b == 0) return a;
  return binary_gcd_odd(a, b >> __builtin_ctzll(b));
}

static void mont_init(mont_t *mo, uint64_t n)
{
  mo->n = n;
  mo->nr = mod_inv_2_64(n);
//  mo->r2 = reminder((unsigned __int128)0 - n, n);
}

static uint64_t from_mont(const mont_t *mo, uint64_t t)
{
  assert(t < mo->n);
  if (t == 0) return 0;
  uint64_t m = (uint64_t)t * mo->nr;
  m = (uint64_t)((unsigned __int128)m * mo->n >> 64);
  return mo->n - m;
}

/* modular powering of (x + y√d) */
static uint64_t pp1(const mont_t *mo, uint64_t p, uint64_t pw)
{
  const uint64_t mod = mo->n;
  uint64_t r2 = reminder((unsigned __int128)0 - mod, mod);

  uint64_t p_one = (0ull - mod) % mod;

  uint64_t d = mont_mul(mo, r2, p * (p - 2));
  uint64_t a = mont_mul(mo, r2, p - 1);
  uint64_t prb = 1ull << (63 - __builtin_clzll(pw));
  uint64_t rx = a, ry = p_one; /* start point */

  while ((prb >>= 1) != 0)
  {
      /* y' = 2*x*y */
      ry = mont_mul(mo, rx, ry);
      ry = addmod(ry, ry, mod);

      /* x' = 2*x*x-1 */
      rx = mont_mul(mo, rx, rx);
      rx = addmod(rx, rx, mod);
      rx = submod(rx, p_one, mod);

      if (pw & prb)
      {
          uint64_t t = addmod(mont_mul(mo, a, rx), mont_mul(mo, ry, d), mod);
          ry = addmod(mont_mul(mo, a, ry), rx, mod);
          rx = t;
      }
  }
  return from_mont(mo, rx);
}

/* modular powering of (x + y√2) */
static uint64_t pp2(const mont_t *mo, uint64_t pw)
{
  const uint64_t mod = mo->n;
  uint64_t prb = 1ull << (63 - __builtin_clzll(pw));

  uint64_t p_one = (0ull - mod) % mod;
  /* start point (3, 2) */
  uint64_t ry = mul2mod(p_one, mod);    
  uint64_t rx = addmod(ry, p_one, mod);

  while ((prb >>= 1) != 0)
  {
      ry = mul2mod(mont_mul(mo, rx, ry), mod);
      rx = mul2mod(mont_mul(mo, rx, rx), mod);
      rx = submod(rx, p_one, mod);
      if (pw & prb)
      {
          uint64_t t = addmod(addmod(mul2mod(rx, mod), rx, mod), mul2mod(mul2mod(ry, mod), mod), mod);
          ry = addmod(addmod(mul2mod(ry, mod), ry, mod), mul2mod(rx, mod), mod);
          rx = t;
      }
  }
  return from_mont(mo, rx);
}

static int lukastest(const mont_t *mo)
{
  const uint64_t n = mo->n;
  uint64_t pw = (n & 3) == 3 ? (n + 1) / 4 : (n + 1) / 2;
  uint64_t rx;

  /* check for n % 8 in 3, 5 */
  if ((n ^ (n >> 1)) & 2)
     rx = pp2(mo, pw);
  else
  {
     uint64_t p = min_nonres(n);
     if (p == 0)
        return 0;
     rx = pp1(mo, p, pw);
  }
  return rx == ((n & 3) == 3 ? 0 : n - 1);
}

static inline uint64_t splitmix64(uint64_t *s)
{
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/* ------------------------------------------------------------------ */
/* Brent's variant of Pollard rho                                      */
/* ------------------------------------------------------------------ */

#define BRENT_BATCH 128
#define BRENT_ATTEMPTS 32
#define BRENT_STEP_LIMIT (1u << 22)
/* Return a non-trivial divisor or 0 if this attempt failed. */
static uint64_t brent_find(const mont_t *mo, uint64_t y0, uint64_t c0)
{
  const uint64_t n = mo->n;
  const uint64_t c = c0;
  uint64_t y = y0;
  uint64_t x = y, ys = y;
  uint64_t q = 1, g = 1, r = 1;
  uint64_t steps = 0;

  while (g == 1) {
    x = y;
    for (uint64_t i = 0; i < r; ++i)
      y = mont_add(mont_mul(mo, y, y), c, n);
    steps += r;

    uint64_t k = 0;
    while (k < r && g == 1) {
      ys = y;
      uint64_t lim = r - k;
      if (lim > BRENT_BATCH) lim = BRENT_BATCH;

      for (uint64_t i = 0; i < lim; ++i) {
        y = mont_add(mont_mul(mo, y, y), c, n);
        uint64_t d = x > y ? x - y : y - x;
        q = mont_mul(mo, q, d);
      }
      steps += lim;

      /* q may have accumulated all prime factors of n */
      g = (q == 0) ? n : gcd_odd(n, q);
      k += lim;
    }

    if (g != 1) break;
    if (steps > BRENT_STEP_LIMIT) return 0;
    r <<= 1;
  }

  if (g == n) {
    /* Find the individual difference that shares a factor with n.
       Before the last batch gcd(q,n) was 1, so the factor is in it. */
    g = 1;
    do {
      ys = mont_add(mont_mul(mo, ys, ys), c, n);
      uint64_t d = x > ys ? x - ys : ys - x;
      if (d == 0) { g = n; break; }
      g = gcd_odd(n, d);
    } while (g == 1);
  }

  return (g > 1 && g < n) ? g : 0;
}

/* ------------------------------------------------------------------ */
/* original Floyd cycle implementation, fallback                       */
/* ------------------------------------------------------------------ */
static uint64_t pollard_floyd(uint64_t m)
{
  uint64_t a;
  for (a = 1; a < m - 1; ++a) {
    uint64_t power = 1, lam = 1;
    uint64_t tortoise = a;
    uint64_t hare = addmod(mulmod(a, a, m), a, m);
    uint64_t prod = 1;
    while (1) {
      uint64_t t = mulmod(submod(tortoise, hare, m), prod, m);
      if (t == 0) {
        t = binary_gcd_odd1(m, prod);
        if (t != 1)
          return t;
        break;
      }
      prod = t;
      if (power == lam) {
        t = binary_gcd_odd1(m, prod);
        if (t != 1)
          return t;
        tortoise = hare;
        power *= 2;
        lam = 0;
      }
      hare = addmod(mulmod(hare, hare, m), a, m);
      lam += 1;
    }
  }
  return 0;
}

/* Return a divisor of the odd composite mo->n, or 0 if all attempts
   failed.  Brent's variant is tried first, Floyd is the fallback. */
static uint64_t pollard(const mont_t *mo)
{
  const uint64_t m = mo->n;
  assert (m % 2 == 1);

  uint64_t r2 = reminder((unsigned __int128)0 - m, m);

  uint64_t state = m ^ 0x9E3779B97F4A7C15ULL;
  for (int att = 0; att < BRENT_ATTEMPTS; ++att) {
    uint64_t y0, c0;
    if (att == 0) {
      y0 = 2 % m;
      c0 = 1;
    } else {
      y0 = 1 + splitmix64(&state) % (m - 1);
      c0 = 1 + splitmix64(&state) % (m - 1);
      if (c0 + 2 == m)      /* f has a fixed point for c = -2 */
        c0 = 1;
    }
    uint64_t g = brent_find(mo, mont_mul(mo, y0, r2), mont_mul(mo, c0, r2));
    if (g != 0)
      return g;
  }

  return pollard_floyd(m);
}

/* Strong Miller-Rabin probable-prime test to base 2. */
static int mont_miller_rabin(const mont_t *mo)
{
  const uint64_t m = mo->n;
  assert(m % 2 == 1);
  uint64_t m1 = m - 1;
  unsigned k = __builtin_ctzll(m1);
  uint64_t power = m1 >> k;

  uint64_t p_one = (0ull - m) % m;
  uint64_t m_one = m - p_one;

  uint64_t result = mul2mod(p_one, m);
  uint64_t prb = 1ull << (63 - __builtin_clzll(power));
  while ((prb >>= 1) != 0)
  {
      result = mont_mul(mo, result, result);

      if (power & prb)
         result = mul2mod(result, m);
  }

  if (result == p_one || result == m_one)
    return 1;

  while (--k)
  {
     result = mont_mul(mo, result, result);
     if (result == p_one)
       return 0;
     if (result == m_one)
       return 1;
  }
  return 0;
}

static uint64_t gcd64(uint64_t a, uint64_t b)
{
    if (a == 0) return b;
    if (b == 0) return a;

    unsigned s = __builtin_ctzll(a | b);
    a >>= __builtin_ctzll(a);
    b >>= __builtin_ctzll(b);

    return binary_gcd_odd(a, b) << s;
}

static u128 pow4_128(uint64_t x)
{
    u128 t = (u128)x * x;
    return t * t;
}

/* index of n (gcd(n,30)=1):  n/30*8 + (n%30)*4/15 */
static inline size_t spf_index(uint64_t n)
{
    return (size_t)(n / 30) * 8 + (size_t)((n % 30) * 4 / 15);
}

/*
 * Look up a divisor in the precomputed smallest-prime-factor table.
 * Returns a prime divisor d (1 < d < n), n itself if n is prime, or
 * 0 when n > FACTOR_TABLE_LIMIT and must be handled by the primality
 * tests and Pollard rho.
 *
 * Keep this out of line: inlining it into process_odd bloats the
 * factorization hot path and costs about 10% on long runs.
 */
static __attribute__((noinline)) uint64_t find_small_prime_divider(uint64_t n)
{
    if (n <= FACTOR_TABLE_LIMIT) {
        uint16_t p = spf_table[spf_index(n)];
        if (p == 0) return n;
        return p;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* factorisation and good / bad prime classification                  */
/* ------------------------------------------------------------------ */
static int add_pair(factor_t *pairs, int *ng, uint64_t p, unsigned e, uint64_t *q)
{
    if ((p & 7) != 1) {
        if (e & 3) return 0;            /* not a fourth power */

        unsigned k = e / 4;
        uint64_t pp = *q * p;
        for (unsigned j = 1; j < k; ++j) {
            pp *= p;
        }
        *q = pp;
        return 1;
    }
    int nk = *ng;
    for (int i = 0; i < nk; ++i) {
        if (pairs[i].p == p) {
            pairs[i].e += e;
            return 1;
        }
    }
    assert(nk < MAX_FACTORS);
    pairs[nk].p = p;
    pairs[nk].e = e;
    *ng = nk + 1;
    return 1;
}

/*
 * Merge prime p with exponent e into the local factor table.
 * Returns the index of p.
 */
static int merge_factor(factor_t *f, int *nf, uint64_t p, unsigned e)
{
    for (int i = 0; i < *nf; ++i) {
        if (f[i].p == p) {
            f[i].e += e;
            return i;
        }
    }
    assert(*nf < MAX_FACTORS);
    f[*nf].p = p;
    f[*nf].e = e;
    return (*nf)++;
}

/* Can the bad prime p still occur in an unprocessed part?  If not, its
   exponent is final and the candidate can be rejected immediately. */
static int p_still_possible(uint64_t p, uint64_t n,
                            const pending_t *stack, int sp)
{
    if (n % p == 0) return 1;
    for (int i = 0; i < sp; ++i)
        if (stack[i].v % p == 0) return 1;
    return 0;
}

/* Strip small prime factors <= 2039 up front.  Every later part is a
   divisor of the stripped number, so these factors cannot reappear;
   if n drops below FACTOR_TABLE_LIMIT the SPF table finishes it.
   This keeps trial division out of the main factorization loop. */
static __attribute__((noinline)) unsigned small_odd_divider(uint64_t *n, factor_t *found)
{
  int fi = 0;
  for (unsigned i = 0; i < CNT(small_prime); ++i)
  {
     /* if (n % small_prime[i] == 0) */
     if (*n * small_prime_inv[i] <= small_prime_lim[i]) {
        unsigned k = 0;
        do {
            *n = *n * small_prime_inv[i];
            ++k;
        } while (*n * small_prime_inv[i] <= small_prime_lim[i]);

        if ((k & 3) && (small_prime[i] & 7) != 1)
            return -1;

        assert(fi < MAX_FACTORS);
        found[fi].p = small_prime[i];
        found[fi].e = k;
        ++fi;
        if (*n < FACTOR_TABLE_LIMIT) break;
     }
  }
  return fi;
}

/*
 * Factor n (odd) and separate its primes into
 *
 *   p = 1 (mod 8)   -> good list (used to build Rg)
 *   p != 1 (mod 8)  -> bad; exponent must be a multiple of 4,
 *                      then p^(e/4) goes into q.
 *
 * No recursion: pollard may return a composite divisor, so such parts
 * are put on an explicit stack.  Prime exponents are accumulated in a
 * local table and checked as soon as no unprocessed part can contain p.
 */
static int process_odd(uint64_t n, uint64_t *q, factor_t *pairs, int *idx)
{
    pending_t stack[MAX_FACTORS];
    int sp = 0;
    unsigned mult = 1;

    factor_t found[MAX_FACTORS];
    int nf = small_odd_divider(&n, found);
    if (nf < 0) return 0;

    for (;;) {
        while (n > 1) {
            uint64_t d = find_small_prime_divider(n);
            int fi;

            if (d == n) {               /* n is prime */
                fi = merge_factor(found, &nf, n, mult);
                n = 1;
            } else if (d != 0) {        /* small prime divisor */
                unsigned k = 0;
                do {
                    n /= d;
                    ++k;
                } while (n % d == 0);

                fi = merge_factor(found, &nf, d, k * mult);
            } else {                    /* n > FACTOR_TABLE_LIMIT */
                /* Primality testing and Pollard rho share one
                   Montgomery context for the same n. */
                mont_t mo;
                mont_init(&mo, n);

                if (mont_miller_rabin(&mo) && lukastest(&mo)) {
                    fi = merge_factor(found, &nf, n, mult);
                    n = 1;
                } else {
                    d = pollard(&mo);
#ifdef USE_COUNTERS
                    ++pollard_counter;
#endif
                    if (d <= 1 || d >= n) return 0;

                    /* exact power of d in n; factor d first so a bad
                       divisor stops the candidate immediately */
                    unsigned k = 0;
                    do {
                        n /= d;
                        ++k;
                    } while (n % d == 0);

                    if (n > 1) {
                        assert(sp < MAX_FACTORS);
                        stack[sp].v = n;
                        stack[sp].mult = mult;
                        ++sp;
                    }
                    n = d;
                    mult *= k;
                    continue;
                }
            }

            if ((found[fi].p & 7) != 1 && (found[fi].e & 3) &&
                !p_still_possible(found[fi].p, n, stack, sp))
                return 0;               /* bad divisor, early exit */
        }

        if (sp == 0) break;

        --sp;
        n = stack[sp].v;
        mult = stack[sp].mult;
    }

    for (int i = 0; i < nf; ++i)
        if (!add_pair(pairs, idx, found[i].p, found[i].e, q))
            return 0;

    return 1;
}

/* ------------------------------------------------------------------ */
/* smallest-prime-factor table construction                           */
/* ------------------------------------------------------------------ */

static int build_factor_table(void)
{
    size_t count = ((size_t)FACTOR_TABLE_LIMIT / 15 + 1) * 4;
    spf_table = calloc(count, sizeof(uint16_t));
    if (!spf_table) return 0;

    const uint32_t L = FACTOR_TABLE_LIMIT;

    for (uint32_t p = 7; (uint64_t)p * p <= L; p += 2) {
        if (p % 3 == 0 || p % 5 == 0) continue;
        if (spf_table[spf_index(p)] != 0) continue;   /* composite */

        for (uint64_t k = p; k <= L / p; k += 2) {
            if (k % 3 == 0 || k % 5 == 0) continue;
            uint64_t m = (uint64_t)p * k;
            size_t idx = spf_index(m);
            if (spf_table[idx] == 0)
                spf_table[idx] = (uint16_t)p;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* residue table                                                      */
/* ------------------------------------------------------------------ */

static int residue_count;
static void add_residue(uint32_t r)
{
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 367 * 625) % PERIOD);
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 1391 * 625) % PERIOD);
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 2415 * 625) % PERIOD);
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 13969 * 625) % PERIOD);
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 14993 * 625) % PERIOD);
    residue_table[residue_count++] = (uint32_t)(((uint64_t)r * (14 * 16384) + 16017 * 625) % PERIOD);
}

static void make_residue_table(void)
{
    residue_count = 0;

    /* all multiples of 5 modulo 625 */
    for (unsigned i = 0; i < 125; ++i)
        add_residue(5 * i);

    /* the four fourth roots of unity modulo 625 */
    add_residue(1);
    add_residue(624);
    add_residue(182);
    add_residue(443);
}

/* ------------------------------------------------------------------ */
/* reporting                                                          */
/* ------------------------------------------------------------------ */

static int report_solution(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
    if (a > b) { uint64_t t = a; a = b; b = t; }

    if (gcd64(gcd64(a, b), gcd64(c, d)) != 1) return 0;

    /* Subtract c^4 from d^4 instead of adding a^4 + b^4 + c^4: for d
       close to 2^32 the sum can wrap around modulo 2^128, while d^4
       itself still fits in u128. */
    u128 d4 = pow4_128(d);
    u128 c4 = pow4_128(c);
    if (c4 >= d4) return 0;

    u128 rest = d4 - c4;
    u128 a4 = pow4_128(a);
    if (a4 >= rest) return 0;
    if (rest - a4 != pow4_128(b)) return 0;

    int seen = 0;
    /* Several threads may reach this point simultaneously. */
#ifdef _OPENMP
#pragma omp critical(solution)
#endif
    {
        int stored = nsol < MAX_SOL ? nsol : MAX_SOL;
        for (int i = 0; i < stored; ++i)
            if (sol_a[i] == a && sol_b[i] == b &&
                sol_c[i] == c && sol_d[i] == d)
                seen = 1;

        if (!seen) {
            if (nsol < MAX_SOL) {
                sol_a[nsol] = a; sol_b[nsol] = b;
                sol_c[nsol] = c; sol_d[nsol] = d;
            }
            ++nsol;

        }
    }

#ifdef _OPENMP
#pragma omp critical(output)
#endif
    {
        printf(seen ? "DUPLICATE: a=%llu b=%llu c=%llu d=%llu\n" : "SOLUTION: a=%llu b=%llu c=%llu d=%llu\n",
               (unsigned long long)a, (unsigned long long)b,
               (unsigned long long)c, (unsigned long long)d);
        fflush(stdout);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* reconstruction of a^4 + b^4 = Rg                                   */
/* ------------------------------------------------------------------ */

static gauss_t gmul(gauss_t x, gauss_t y)
{
    __int128 re = (__int128)x.re * y.re - (__int128)x.im * y.im;
    __int128 im = (__int128)x.re * y.im + (__int128)x.im * y.re;
    gauss_t z = { (int64_t)re, (int64_t)im };
    return z;
}

/* p = A^2 + B^2, p prime, p = 1 (mod 8) */
static void gauss_repr_raw(uint64_t p, int64_t *A, int64_t *B)
{
    unsigned g = min_nonres(p);
    /* For a prime p = 1 (mod 8) a quadratic non-residue exists and
       is_square_odd(p) is false, so min_nonres cannot return 0. */
    assert(g != 0);

    uint64_t z = powmod(g, (p - 1) / 4, p);   /* z^2 = -1 (mod p) */
    uint64_t a = p, b = z;

    while ((u128)b * b > p) {
        uint64_t r = a % b;
        a = b;
        b = r;
    }

    uint64_t x = b;
    uint64_t y2 = p - x * x;
    uint32_t y = is_square(y2);
    /* Cornacchia's descent guarantees x != 0 and that p - x^2 is a square. */
    assert(y != 0 && x != 0);
    assert(x * x + (uint64_t)y * y == p);

    *A = (int64_t)x;
    *B = (int64_t)y;
}

static inline size_t gauss_pre_hash(uint64_t p)
{
    return (size_t)((p * 0x9E3779B97F4A7C15ULL) >> 32) & gauss_pre_mask;
}

static int gauss_pre_lookup(uint64_t p, int64_t *A, int64_t *B)
{
    if (!gauss_pre || p > GAUSS_PRECOMP_LIMIT) return 0;

    size_t h = gauss_pre_hash(p);
    while (gauss_pre[h].p != 0) {
        if (gauss_pre[h].p == p) {
            *A = gauss_pre[h].A;
            *B = gauss_pre[h].B;
            return 1;
        }
        h = (h + 1) & gauss_pre_mask;
    }
    return 0;
}

static void build_gauss_precomp(void)
{
    /* For p coprime to 30 the SPF table is an exact primality test up to
       FACTOR_TABLE_LIMIT, so no separate primality check is needed. */
    uint32_t count = 0;
    for (uint32_t p = 17; p <= GAUSS_PRECOMP_LIMIT; p += 8)
        if (p % 3 && p % 5 && spf_table[spf_index(p)] == 0)
            ++count;

    size_t size = 1;
    while (size < (size_t)count * 2) size <<= 1;
    gauss_pre = calloc(size, sizeof(*gauss_pre));
    if (!gauss_pre) {
        fprintf(stderr, "warning: cannot allocate Gaussian table, "
                        "computing representations on demand\n");
        return;
    }
    gauss_pre_mask = size - 1;

    for (uint32_t p = 17; p <= GAUSS_PRECOMP_LIMIT; p += 8) {
        if (p % 3 == 0 || p % 5 == 0) continue;
        if (spf_table[spf_index(p)] != 0) continue;

        int64_t A, B;
        gauss_repr_raw(p, &A, &B);

        size_t h = gauss_pre_hash(p);
        while (gauss_pre[h].p != 0) h = (h + 1) & gauss_pre_mask;
        gauss_pre[h].p = p;
        gauss_pre[h].A = (uint32_t)A;
        gauss_pre[h].B = (uint32_t)B;
    }
}

/* p = A^2 + B^2, p prime, p = 1 (mod 8); use the precomputed table
   when possible. */
static void gauss_prime_repr(uint64_t p, int64_t *A, int64_t *B)
{
    if (gauss_pre_lookup(p, A, B)) return;
    gauss_repr_raw(p, A, B);
}

static int enum_gauss(enum_ctx *ctx, int idx, gauss_t cur)
{
    if (ctx->found) return 1;

    if (idx == ctx->ng) {
        gauss_t z = cur;
        if (ctx->delta2) {
            gauss_t w = { 1, 1 };
            z = gmul(z, w);
        }

        int64_t re = z.re < 0 ? -z.re : z.re;
        int64_t im = z.im < 0 ? -z.im : z.im;

        uint64_t u = (uint64_t)re;
        uint64_t v = (uint64_t)im;
        if (u == 0 || v == 0) return 0;
        /* The norm is ctx->R (the Gaussian product Rg) by construction;
           a mismatch would be a bug. */
        assert((u128)u * u + (u128)v * v == ctx->R);

        uint32_t ap = is_square(u);
        uint32_t bp = is_square(v);
        if (!ap || !bp) return 0;

        /* q starts at 1 and only grows. */
        assert(ctx->q != 0);
        if (ap > UINT64_MAX / ctx->q ||
            bp > UINT64_MAX / ctx->q)
            return 0;

        uint64_t a = ctx->q * ap;
        uint64_t b = ctx->q * bp;
        if (a >= ctx->d || b >= ctx->d) return 0;

        if (!report_solution(a, b, ctx->c, ctx->d)) return 0;
        ctx->found = 1;
        return 1;
    }

    uint64_t p = ctx->good[idx].p;
    unsigned e = ctx->good[idx].e;

    int64_t A, B;
    gauss_prime_repr(p, &A, &B);

    gauss_t pi = { A, B };
    gauss_t bar = { A, -B };

    /* powers pi^k and bar^(e-k) */
    gauss_t pp[65], bb[65];
    pp[0] = (gauss_t){1, 0};
    bb[0] = (gauss_t){1, 0};
    for (unsigned k = 1; k <= e; ++k) {
        pp[k] = gmul(pp[k-1], pi);
        bb[k] = gmul(bb[k-1], bar);
    }

    unsigned kmin = 0;
    unsigned kmax = e;
    if (idx == 0) kmax = e / 2;   /* remove global conjugation duplicates */

    for (unsigned k = kmin; k <= kmax; ++k) {
        gauss_t f = gmul(pp[k], bb[e - k]);
        if (enum_gauss(ctx, idx + 1, gmul(cur, f))) return 1;
    }
    return 0;
}

static int try_gauss(u128 R, factor_t *good, int ng, int delta2,
                     uint64_t q, uint64_t c, uint64_t d)
{
    enum_ctx ctx;
    ctx.good = good;
    ctx.ng = ng;
    ctx.delta2 = delta2;
    ctx.R = R;
    ctx.q = q;
    ctx.c = c;
    ctx.d = d;
    ctx.found = 0;

    gauss_t one = { 1, 0 };
    enum_gauss(&ctx, 0, one);
    return ctx.found;
}

#ifdef USE_COUNTERS
uint64_t counters[32] = {0};
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define FORBIDDEN13 ((1u << 7) | (1u << 8) | (1u << 11))
#define FORBIDDEN29 ((1u << 4) | (1u << 5) | (1u << 6) | (1u << 9) | \
                     (1u << 13) | (1u << 22) | (1u << 28))
/* ------------------------------------------------------------------ */
/* candidate check                                                    */
/* ------------------------------------------------------------------ */
static int check_candidate(uint64_t c, uint64_t d)
{
    /* Note: gcd(c,d) need not be 1.  A common prime p is possible only
       for p = 1 (mod 8); such primes stay in the good list and are kept
       in the Gaussian product Rg below. */
#ifdef USE_COUNTERS
    counters[0]++;
#endif

    uint64_t x = d - c;
    uint64_t y = c + d;
    uint64_t pz = d * d - x * y / 2; /* (c^2 + d^2) / 2 */

    /* powers of two (cheapest check, do it first) */
    unsigned vx = __builtin_ctzll(x);
    unsigned vy = __builtin_ctzll(y);
    unsigned vt = vx + vy + 1;

    if (unlikely((vt & 3) > 1)) return 0;       /* must be 0 or 1 mod 4 */

    /* odd parts after removing only powers of two */
    uint64_t px = x >> vx;
    uint64_t py = y >> vy;

    /* The odd parts of x and y must agree modulo 8. */
    if (unlikely((px ^ py) & 7)) return 0;

    uint32_t delta2 = (uint32_t)(vt & 3);    /* 0 or 1 */

    /* The odd part of a^4+b^4 is 1 mod 16; when delta2 == 0 this
       forces an extra agreement in bit 3. */
    if (unlikely( (px ^ py) & ((delta2 ^ 1) << 3) )) return 0;
#ifdef USE_COUNTERS
    counters[1]++;
#endif

    uint64_t q = (uint64_t)1 << (vt / 4);

    uint64_t *f;

    /* powers of five; this branch covers 4 of the 129 residues */
    if (unlikely(c % 5 != 0)) {
        /* d % 5 != 0.  Then exactly one of x,y,z is divisible by 5.
           For a possible solution all of the 5-adic valuation must sit
           in that factor, so it has to be divisible by 625.
           The residue table guarantees 625 | x*y*z.  Since exactly one
           of x,y,z contains the factor 5, this factor is automatically
           divisible by 625, so no separate %625 check is needed. */
        if (x % 5 == 0)      f = &px;
        else if (y % 5 == 0) f = &py;
        else                 f = &pz;
        uint64_t ff = *f;
        do {
            ff /= 625;
            q *= 5;
        } while (ff % 625 == 0);
        *f = ff;
        uint32_t r5 = (uint32_t)((px << delta2) * py * (pz % 5) % 5);
        if (r5 != 1 && r5 != 2) return 0;
    }
#ifdef USE_COUNTERS
    counters[2]++;
#endif

    /* powers of three: if c,d are both non-zero mod 3, exactly one
       of x,y is divisible by 3 (z is 2 mod 3).  A possible solution
       needs the whole 3-adic part in that factor, hence 81 | it. */
    if (unlikely(x % 3 == 0)) f = &px;
    else if (unlikely(y % 3 == 0)) f = &py;
    else f = NULL;
    if (unlikely(f != NULL)) {
        uint64_t ff = *f;
        do {
            ff /= 81;
            q *= 3;
        } while (ff % 81 == 0);

        if (ff % 3 == 0) return 0;
        *f = ff;
    }
#ifdef USE_COUNTERS
    counters[3]++;
#endif

    u128 R = (px << delta2) * py * (u128)pz;

    /* 128-bit division is slow, so compute the remainder once for
       several moduli. */
    uint64_t rem = reminder(R, (uint64_t)7 * 7 * 7 * 7 * 11 * 11 * 11 * 11 * 13 * 13 * 13 * 13 * 29 * 29 * 29 * 29);

    /* A fourth power cannot be 7, 8 or 11 (mod 13), nor any residue in
       FORBIDDEN29 modulo 29. */
    uint32_t r29 = (uint32_t)(rem % 29);
    if (unlikely(FORBIDDEN29 & (1u << r29))) return 0;
#ifdef USE_COUNTERS
    counters[4]++;
#endif

    uint32_t r13 = (uint32_t)(rem % 13);
    if (unlikely(FORBIDDEN13 & (1u << r13))) return 0;
#ifdef USE_COUNTERS
    counters[5]++;
#endif

    if (r13 == 0 && rem % (13 * 13 * 13 * 13) != 0) return 0;
#ifdef USE_COUNTERS
    counters[6]++;
#endif

    if (rem % 7 == 0 && rem % (7 * 7 * 7 * 7) != 0) return 0;
#ifdef USE_COUNTERS
    counters[7]++;
#endif

    if (r29 == 0 && rem % (29 * 29 * 29 * 29) != 0) return 0;
#ifdef USE_COUNTERS
    counters[8]++;
#endif

    if (rem % 11 == 0 && rem % (11 * 11 * 11 * 11) != 0) return 0;
#ifdef USE_COUNTERS
    counters[9]++;
#endif

    factor_t good[MAX_FACTORS];
    int ng = 0;

    if (!process_odd((px < py) ? px : py, &q, good, &ng)) return 0;
    if (!process_odd((px < py) ? py : px, &q, good, &ng)) return 0;
    if (!process_odd(pz, &q, good, &ng)) return 0;
#ifdef USE_COUNTERS
    counters[10]++;
#endif

    /* The Gaussian product contains only primes p = 1 (mod 8) and the
       leftover factor 2^delta2.  Bad primes were already moved into q;
       leaving them in Rg would make u^2 + v^2 = Rg unsolvable. */
    u128 Rg = (u128)(delta2 + 1);
    for (int i = 0; i < ng; ++i) {
        uint64_t p = good[i].p;
        for (unsigned e = good[i].e; e != 0; --e)
            Rg *= p;
    }

    return try_gauss(Rg, good, ng, delta2, q, c, d);
}

static void show_usage(char* name) {
    fprintf(stderr, "usage: %s [end] or %s <start> <end>\n", name, name);
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    uint64_t limit = 3000000;
    uint64_t start = 1;

    if (argc > 3) {
        show_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        char *end = NULL;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || v == 0) {
            show_usage(argv[0]);
            return EXIT_FAILURE;
        }
        limit = (uint64_t)v;
    } else if (argc == 3) {
        char *end = NULL;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || v == 0) {
            show_usage(argv[0]);
            return EXIT_FAILURE;
        }
        start = (uint64_t)v;

        end = NULL;
        v = strtoull(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || v == 0) {
            show_usage(argv[0]);
            return EXIT_FAILURE;
        }
        limit = (uint64_t)v;
    }

    if (limit > MAX_D_LIMIT) limit = MAX_D_LIMIT;

    while ((start & 7) != 1) ++start;
    if (start > limit) {
        printf("empty range: no d = 1 (mod 8) in [%llu, %llu]\n",
               (unsigned long long)start, (unsigned long long)limit);
        return 0;
    }

    make_residue_table();

    if (!build_factor_table()) {
        fprintf(stderr, "out of memory building factor table\n");
        return EXIT_FAILURE;
    }

    build_gauss_precomp();

    printf("searching d in [%llu, %llu]\n",
           (unsigned long long)start, (unsigned long long)limit);
    fflush(stdout);

    uint64_t cand = 0;
    time_t t0 = time(NULL);
#ifndef _OPENMP
    uint64_t next_report = start + 100000;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 25) reduction(+:cand)
#endif
    for (uint64_t d = start; d <= limit; d += 8) {
        if (d % 5 == 0) continue;

#ifndef _OPENMP
        if (d >= next_report) {
            fprintf(stderr, "d=%llu candidates=%llu elapsed=%lds\n",
                    (unsigned long long)d, (unsigned long long)cand,
                    (long)(time(NULL) - t0));
            next_report += 100000;
        }
#else
        /* rough progress from the first worker */
        if (omp_get_thread_num() == 0 && d >= 100000 && (d % 100000) < 8) {
#pragma omp critical(output)
            fprintf(stderr, "d=%llu elapsed=%lds\n",
                    (unsigned long long)d, (long)(time(NULL) - t0));
        }
#endif

        uint64_t dm = d % PERIOD;

        for (size_t i = 0; i < CNT(residue_table); ++i) {
            uint64_t c = dm * residue_table[i] % PERIOD;
            if (d % 3 == 0) {
                uint64_t rc = c % 3;
                if (rc == 2) {
                    if (c < d) {
                        ++cand;
                        check_candidate(c, d);
                    }
                    c += 2 * PERIOD;
                }
                else if (rc == 0)
                    c += PERIOD;
                /* now c % 3 == 1 */
                while (c < d) {
                    ++cand;
                    check_candidate(c, d);
                    c += PERIOD;
                    if (c < d) {
                        ++cand;
                        check_candidate(c, d);
                    }
                    else break;
                    c += 2 * PERIOD;
                }
            } else { /* d % 3 != 0 */
                uint64_t cc = c;
                uint64_t rc = c % 3;
                if (rc != 0) c += (3 - rc) * PERIOD;
                while (c < d) {
                    ++cand;
                    check_candidate(c, d);
                    c += 3 * PERIOD;
                }
                c = cc + (d % 81 + 81 - cc % 81) * 4 % 81 * PERIOD;
                /* (d - c) % 81 == 0 */
                while (c < d) {
                    ++cand;
                    check_candidate(c, d);
                    c += 81 * PERIOD;
                }
                c = cc + (162 - d % 81 - cc % 81) * 4 % 81 * PERIOD;
                /* (d + c) % 81 == 0 */
                while (c < d) {
                    ++cand;
                    check_candidate(c, d);
                    c += 81 * PERIOD;
                }
            }
        }
    }

    printf("done: %llu candidates, %d solution(s), %lds\n",
           (unsigned long long)cand, nsol, (long)(time(NULL) - t0));
#ifdef USE_COUNTERS
    printf("filters: %llu ", (unsigned long long)counters[0]);
    for(unsigned i = 1; counters[i] != 0; ++i)
        printf("(%u) %g %llu ", i, (double)counters[i]/counters[i-1],
               (unsigned long long)counters[i]);
    puts("");
    printf("pollard calls %llu\n", (unsigned long long)pollard_counter);
#endif
    free(spf_table);
    free(gauss_pre);
    return 0;
}
