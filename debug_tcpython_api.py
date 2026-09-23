# debug_tcpython_api.py
#
# Thermo-Calc TC-Python에서
# BCC_A2 phase fraction 및 BCC_A2 내부 Cr 함량을
# 어떤 quantity 문법으로 읽어야 하는지 확인하는 최소 진단 코드.
#
# 출력이 너무 길어지지 않도록 필요한 항목만 출력한다.

from pathlib import Path
import os

from tc_python import TCPython, ThermodynamicQuantity

from config import (
    DATABASE,
    ELEMENTS,
    BASE_WT,
    T_EQUIL_K,
    P_PA,
)


def get_tc_cache_folder():
    cache = (
        Path(os.environ["LOCALAPPDATA"])
        / "ThermoCalcCache"
        / "poseidon500_debug"
    )

    cache.mkdir(parents=True, exist_ok=True)

    return cache


def show_test(label, query_function, result):
    """
    query_function으로 ThermodynamicQuantity를 만들고,
    result.get_value_of() 실행 결과 또는 오류를 짧게 출력한다.
    """
    try:
        quantity = query_function()
        value = result.get_value_of(quantity)

        print(f"[SUCCESS] {label}")
        print(f"          quantity = {quantity}")
        print(f"          value    = {value}")

    except Exception as exc:
        print(f"[FAILED]  {label}")
        print(f"          error = {repr(exc)}")


def main():
    # 대표 조성 하나
    composition = dict(BASE_WT)
    composition["CE"] = 0.010
    composition["CR"] = 1.000

    print("=" * 80)
    print("TC-Python Minimal Quantity Debug")
    print("=" * 80)
    print(f"Database: {DATABASE}")
    print(f"Composition: {composition}")
    print(f"Temperature: {T_EQUIL_K - 273.15:.1f} °C")
    print("=" * 80)

    with TCPython() as start:
        start.set_cache_folder(str(get_tc_cache_folder()))

        system = (
            start
            .select_database_and_elements(
                DATABASE,
                ELEMENTS
            )
            .get_system()
        )

        calc = (
            system
            .with_single_equilibrium_calculation()
            .set_condition(
                ThermodynamicQuantity.temperature(),
                T_EQUIL_K
            )
            .set_condition(
                ThermodynamicQuantity.pressure(),
                P_PA
            )
        )

        for element, wt_percent in composition.items():
            calc = calc.set_condition(
                ThermodynamicQuantity.mass_fraction_of_a_component(
                    element
                ),
                float(wt_percent) / 100.0
            )

        result = calc.calculate()

        print("\n[1] Stable phase instances returned by result:")
        print("-" * 80)

        stable_phases = list(result.get_stable_phases())

        for phase in stable_phases:
            print(f"  {phase}")

        print("\n[2] ThermodynamicQuantity methods containing 'phase' or 'fraction':")
        print("-" * 80)

        keywords = ["phase", "fraction", "component", "mass"]

        methods = [
            method
            for method in dir(ThermodynamicQuantity)
            if not method.startswith("_")
            and any(key in method.lower() for key in keywords)
        ]

        for method in methods:
            print(f"  {method}")

        print("\n[3] Phase fraction query tests:")
        print("-" * 80)

        # A. 가장 가능성이 큰 ThermodynamicQuantity helper
        show_test(
            "phase_fraction('BCC_A2')",
            lambda: ThermodynamicQuantity.phase_fraction(
                "BCC_A2"
            ),
            result
        )

        # B. phase instance 이름으로 query
        show_test(
            "phase_fraction('BCC_A2#1')",
            lambda: ThermodynamicQuantity.phase_fraction(
                "BCC_A2#1"
            ),
            result
        )

        # C. Generic quantity 명령문 테스트
        if hasattr(ThermodynamicQuantity, "user_defined"):
            show_test(
                "user_defined('NP(BCC_A2)')",
                lambda: ThermodynamicQuantity.user_defined(
                    "NP(BCC_A2)"
                ),
                result
            )

        print("\n[4] Matrix Cr query tests:")
        print("-" * 80)

        # A. component, phase 순서
        show_test(
            "mass_fraction_of_component_in_phase('CR', 'BCC_A2')",
            lambda: (
                ThermodynamicQuantity
                .mass_fraction_of_component_in_phase(
                    "CR",
                    "BCC_A2"
                )
            ),
            result
        )

        # B. phase, component 순서
        show_test(
            "mass_fraction_of_component_in_phase('BCC_A2', 'CR')",
            lambda: (
                ThermodynamicQuantity
                .mass_fraction_of_component_in_phase(
                    "BCC_A2",
                    "CR"
                )
            ),
            result
        )

        # C. instance 이름
        show_test(
            "mass_fraction_of_component_in_phase('CR', 'BCC_A2#1')",
            lambda: (
                ThermodynamicQuantity
                .mass_fraction_of_component_in_phase(
                    "CR",
                    "BCC_A2#1"
                )
            ),
            result
        )

        # D. Generic Console-style expression
        if hasattr(ThermodynamicQuantity, "user_defined"):
            show_test(
                "user_defined('W(CR,BCC_A2)')",
                lambda: ThermodynamicQuantity.user_defined(
                    "W(CR,BCC_A2)"
                ),
                result
            )

        print("\n[5] Debug complete.")
        print("=" * 80)


if __name__ == "__main__":
    main()