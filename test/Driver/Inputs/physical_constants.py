"""Independent 80-digit Decimal formulas, with declared f64 error budgets.

Read defining values from the one public header; do not copy them here.
The largest multiplicative chain (Debye through Coulomb and R) has at most
24 input-rounding/arithmetic factors. gamma_n = n*u/(1-n*u) bounds their
relative error. Counts include repeated inputs and inverse factors, and
are deliberately conservative for exact powers of two. This checks the
compiler-rounded derivation, not equality to a second rounded literal.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import re
import struct
import subprocess
import sys

header=Path(sys.argv[1]).read_text()
def defining(name):
    return Decimal(re.search(r"^#define MDIR_PHYSICAL_"+name+r" ([0-9.e+-]+)$",header,re.M)[1])

with localcontext() as context:
    context.prec=80
    # Gauss-Legendre arithmetic supplies pi independently of the f64 header.
    a=Decimal(1);b=1/Decimal(2).sqrt();t=Decimal(1)/4;p=Decimal(1)
    for _ in range(7):
        next_a=(a+b)/2
        b=(a*b).sqrt();t-=p*(a-next_a)**2;a=next_a;p*=2
    pi=(a+b)**2/(4*t)
    k=defining("BOLTZMANN_SI");na=defining("AVOGADRO")
    e=defining("ELEMENTARY_CHARGE");eps=defining("VACUUM_PERMITTIVITY")
    kcal=defining("KJ_PER_KCAL");angstrom=defining("NM_PER_ANGSTROM")
    atmosphere=defining("BAR_PER_ATM")
    coulomb=na*e*e/(4*pi*eps)*Decimal(10)**6
    gas=k*na/1000
    pressure=Decimal(10)**25/na
    reference={"coulomb":coulomb,"coulomb_control":coulomb/(kcal*angstrom),
        "gas":gas,"gas_control":gas/kcal,"pressure_bar":pressure,
        "pressure_atm":pressure/atmosphere,"density":Decimal(10)**21/na,
        # Independent SI screening formula, rather than reusing C/R.
        "debye":2*na*e*e/(eps*k)*Decimal(10)**-15,
        "boltzmann":k,"avogadro":na,"charge":e,"permittivity":eps}
    # Coulomb: five repeated-input roundings plus six operations; control
    # conversion adds two inputs and two operations. R needs four factors;
    # pressure has its two prefix inputs, N_A, and three divisions. Debye
    # through C/R adds repeated inputs/operations, covered by 24 factors.
    budgets={"coulomb":12,"coulomb_control":16,"gas":4,"gas_control":6,
        "pressure_bar":6,"pressure_atm":8,"density":3,"debye":24,
        "boltzmann":1,"avogadro":1,"charge":1,"permittivity":1}
    output=subprocess.check_output([sys.argv[2]],text=True)
    measured={}
    for line in output.splitlines():
        name,value,bits=line.split();value=float(value)
        assert struct.unpack("=Q",struct.pack("=d",value))[0]==int(bits),(name,"value lost bits")
        measured[name]=value
    assert measured.keys()==reference.keys()
    unit=Decimal(2)**-53
    for name,expected in reference.items():
        n=budgets[name];gamma=n*unit/(1-n*unit)
        error=abs(Decimal.from_float(measured[name])-expected)
        bound=gamma*abs(expected)
        assert error<=bound,(name,error,bound)
        print(f"{name}: {measured[name]:.17g}; Decimal relative error {error/abs(expected):.4E}, bound {gamma:.4E}")
print("physical constants: Decimal formulas and C/C++ bits passed")
