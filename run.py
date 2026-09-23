# run.py
import config
from tc_python import TCPython, ThermodynamicQuantity

print("Thermo-Calc 세션 시작 중...")

# 1. TCPython 세션 연결
with TCPython() as tc:
    # 2. 데이터베이스 및 원소 로드 (config.py 값 참조)
    system = (
        tc.select_database_and_elements(config.DATABASE, config.ELEMENTS)
        .without_default_phases()
        .select_phase(config.MATRIX_PHASE)
    )
    
    calc = system.get_system().with_single_equilibrium_calculation()

    # 3. 계산 조건 설정
    calc.set_condition(ThermodynamicQuantity.temperature(), config.T_EQUIL_K)
    calc.set_condition(ThermodynamicQuantity.pressure(), config.P_PA)

    # 기본 성분 주입 (wt% -> 질량 분율 변환)
    for elem, val in config.BASE_WT.items():
        calc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(elem), val / 100.0
        )

    # 테스트 성분: Cr, Ce 1개 조합만 먼저 단일점 계산
    calc.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component("CR"), 1.0 / 100.0)
    calc.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component("CE"), 0.01 / 100.0)

    # 4. 평형 계산 수행
    print("평형 계산 실행 중...")
    result = calc.calculate()
    matrix_vol = result.get_value_of(
        ThermodynamicQuantity.volume_fraction_of_a_phase(config.MATRIX_PHASE)
    )
    print(f"계산 완료! {config.MATRIX_PHASE} 분율: {matrix_vol:.4f}")