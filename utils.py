"""
工具函数模块 - 包含 eGFR 计算和药物剂量规则引擎
用于校验 LLM 输出，防止幻觉

Author: Immunosuppressant Advisor Team
"""

import math
from typing import Dict, List, Tuple, Optional


def calculate_egfr(age: int, serum_creatinine: float, is_female: bool = False, is_black: bool = False) -> float:
    """
    使用 CKD-EPI 公式计算估算肾小球滤过率 (eGFR)

    参数:
        age: 患者年龄（岁）
        serum_creatinine: 血清肌酐值 (mg/dL)
        is_female: 是否为女性
        is_black: 是否为黑人

    返回:
        eGFR 值 (mL/min/1.73m²)
    """
    if serum_creatinine <= 0 or age <= 0:
        return 0.0

    serum_creatinine = float(serum_creatinine)

    if is_female:
        if is_black:
            if serum_creatinine <= 0.7:
                egfr = 166 * (serum_creatinine / 0.7) ** -0.099 * (0.993 ** age)
            else:
                egfr = 166 * (serum_creatinine / 0.7) ** -1.20 * (0.993 ** age)
        else:
            if serum_creatinine <= 0.7:
                egfr = 144 * (serum_creatinine / 0.7) ** -0.329 * (0.993 ** age)
            else:
                egfr = 144 * (serum_creatinine / 0.7) ** -1.209 * (0.993 ** age)
    else:
        if is_black:
            if serum_creatinine <= 0.7:
                egfr = 163 * (serum_creatinine / 0.7) ** -0.099 * (0.993 ** age)
            else:
                egfr = 163 * (serum_creatinine / 0.7) ** -1.20 * (0.993 ** age)
        else:
            if serum_creatinine <= 0.7:
                egfr = 141 * (serum_creatinine / 0.7) ** -0.329 * (0.993 ** age)
            else:
                egfr = 141 * (serum_creatinine / 0.7) ** -1.209 * (0.993 ** age)

    return round(egfr, 2)


def get_egfr_stage(egfr: float) -> Tuple[str, str]:
    """
    根据 eGFR 值返回 CKD 分期和描述

    返回:
        (分期名称, 描述)
    """
    if egfr >= 90:
        return "G1", "肾功能正常"
    elif egfr >= 60:
        return "G2", "肾功能轻度下降"
    elif egfr >= 45:
        return "G3a", "肾功能轻至中度下降"
    elif egfr >= 30:
        return "G3b", "肾功能中至重度下降"
    elif egfr >= 15:
        return "G4", "肾功能重度下降"
    else:
        return "G5", "肾衰竭"


def get_drug_info(drug_name: str) -> Dict:
    """
    获取药物基本信息和剂量范围

    参数:
        drug_name: 药物名称（他克莫司/环孢素/霉酚酸酯）

    返回:
        包含药物信息的字典
    """
    drug_info = {
        "他克莫司": {
            "english_name": "Tacrolimus",
            "dose_unit": "mg/day",
            "standard_dose_range": (2, 10),
            "target_concentration": {
                "kidney_transplant": {"min": 5.0, "max": 15.0, "unit": "ng/mL"},
                "liver_transplant": {"min": 5.0, "max": 20.0, "unit": "ng/mL"},
                "heart_transplant": {"min": 8.0, "max": 20.0, "unit": "ng/mL"},
            },
            "initial_dose_per_kg": 0.1,
            "dose_unit_per_kg": "mg/kg/day",
            "therapeutic_range_note": "他克莫司治疗窗窄，个体差异大，需血药浓度监测",
        },
        "环孢素": {
            "english_name": "Cyclosporine",
            "dose_unit": "mg/day",
            "standard_dose_range": (100, 400),
            "target_concentration": {
                "kidney_transplant": {"min": 100, "max": 300, "unit": "ng/mL"},
                "liver_transplant": {"min": 100, "max": 350, "unit": "ng/mL"},
                "heart_transplant": {"min": 150, "max": 350, "unit": "ng/mL"},
            },
            "initial_dose_per_kg": 4.0,
            "dose_unit_per_kg": "mg/kg/day",
            "therapeutic_range_note": "环孢素需监测谷浓度，调整剂量需谨慎",
        },
        "霉酚酸酯": {
            "english_name": "Mycophenolate Mofetil",
            "dose_unit": "mg/day",
            "standard_dose_range": (1000, 2000),
            "target_concentration": {
                "kidney_transplant": {"min": 1.0, "max": 3.5, "unit": "μg/mL"},
                "liver_transplant": {"min": 1.0, "max": 3.5, "unit": "μg/mL"},
                "heart_transplant": {"min": 1.0, "max": 3.5, "unit": "μg/mL"},
            },
            "initial_dose_per_kg": 30.0,
            "dose_unit_per_kg": "mg/kg/day",
            "therapeutic_range_note": "霉酚酸酯通常与钙调磷酸酶抑制剂和糖皮质激素联用",
        },
        "西罗莫司": {
            "english_name": "Sirolimus",
            "dose_unit": "mg/day",
            "standard_dose_range": (1, 5),
            "target_concentration": {
                "kidney_transplant": {"min": 4.0, "max": 12.0, "unit": "ng/mL"},
                "liver_transplant": {"min": 4.0, "max": 12.0, "unit": "ng/mL"},
                "heart_transplant": {"min": 4.0, "max": 12.0, "unit": "ng/mL"},
            },
            "initial_dose_per_kg": 0.05,
            "dose_unit_per_kg": "mg/kg/day",
            "therapeutic_range_note": "西罗莫司为哺乳动物雷帕霉素靶蛋白抑制剂，需监测血药浓度",
        },
        "依维莫司": {
            "english_name": "Everolimus",
            "dose_unit": "mg/day",
            "standard_dose_range": (1, 5),
            "target_concentration": {
                "kidney_transplant": {"min": 3.0, "max": 8.0, "unit": "ng/mL"},
                "liver_transplant": {"min": 3.0, "max": 8.0, "unit": "ng/mL"},
                "heart_transplant": {"min": 3.0, "max": 8.0, "unit": "ng/mL"},
            },
            "initial_dose_per_kg": 0.05,
            "dose_unit_per_kg": "mg/kg/day",
            "therapeutic_range_note": "依维莫司为哺乳动物雷帕霉素靶蛋白抑制剂，可用于肾移植后排斥反应的预防",
        },
    }
    return drug_info.get(drug_name, {})


def calculate_initial_dose(drug_name: str, weight: float, egfr: float, transplant_type: str, concomitant_drugs: Optional[List[str]] = None) -> Dict:
    """
    根据患者信息计算初始剂量

    参数:
        drug_name: 药物名称
        weight: 体重 (kg)
        egfr: 估算肾小球滤过率
        transplant_type: 移植类型
        concomitant_drugs: 联用药物列表（如五酯胶囊）

    返回:
        包含剂量建议的字典
    """
    drug_info = get_drug_info(drug_name)
    if not drug_info:
        return {"error": "未知的药物"}

    initial_dose_per_kg = drug_info["initial_dose_per_kg"]
    calculated_dose = weight * initial_dose_per_kg
    adjustment_notes = []

    if egfr < 30:
        calculated_dose = calculated_dose * 0.75
        adjustment_notes.append("根据eGFR<30mL/min/1.73m²，剂量降低25%")
    elif egfr < 60:
        calculated_dose = calculated_dose * 0.875
        adjustment_notes.append("根据eGFR 30-59mL/min/1.73m²，剂量降低12.5%")
    else:
        adjustment_notes.append("eGFR正常，无需剂量调整")

    if concomitant_drugs:
        wuzhi_drugs = ["五酯胶囊", "五酯软胶囊", "五酯胶囊/软胶囊", "Wuzhi Capsule"]
        for drug in concomitant_drugs:
            if drug in wuzhi_drugs:
                calculated_dose = calculated_dose * 0.7
                adjustment_notes.append("联用五酯胶囊/软胶囊，剂量降低30%（五酯胶囊可升高他克莫司血药浓度）")
                break

    calculated_dose = round(calculated_dose, 1)

    target_range = drug_info["target_concentration"].get(transplant_type, {})

    return {
        "drug_name": drug_name,
        "english_name": drug_info["english_name"],
        "calculated_dose": calculated_dose,
        "dose_unit": drug_info["dose_unit"],
        "dose_per_kg": initial_dose_per_kg,
        "dose_unit_per_kg": drug_info["dose_unit_per_kg"],
        "target_concentration": target_range,
        "adjustment_note": "；".join(adjustment_notes),
        "therapeutic_range_note": drug_info["therapeutic_range_note"],
    }


def interpret_concentration(
    drug_name: str,
    measured_concentration: float,
    transplant_type: str,
    sampling_time: str = "谷浓度"
) -> Dict:
    """
    解读血药浓度检测结果

    参数:
        drug_name: 药物名称
        measured_concentration: 实测浓度
        transplant_type: 移植类型
        sampling_time: 采样时间（谷浓度/峰浓度）

    返回:
        包含解读结果的字典
    """
    drug_info = get_drug_info(drug_name)
    if not drug_info:
        return {"error": "未知的药物"}

    target_range = drug_info["target_concentration"].get(transplant_type, {})
    if not target_range:
        return {"error": "未知的移植类型"}

    min_conc = target_range["min"]
    max_conc = target_range["max"]
    unit = target_range["unit"]

    if measured_concentration < min_conc:
        deficit_percent = round((min_conc - measured_concentration) / min_conc * 100, 1)
        status = "偏低"
        recommendation = f"浓度低于目标范围下限{unit}，建议增加剂量约{deficit_percent}%，具体调整请遵医嘱"
        action = "增加剂量"
    elif measured_concentration > max_conc:
        excess_percent = round((measured_concentration - max_conc) / max_conc * 100, 1)
        status = "偏高"
        recommendation = f"浓度高于目标范围上限{unit}，建议降低剂量约{excess_percent}%，密切监测肾功能，具体调整请遵医嘱"
        action = "降低剂量"
    else:
        status = "达标"
        recommendation = f"浓度在目标范围内（{min_conc}-{max_conc} {unit}），维持当前剂量"
        action = "维持剂量"

    return {
        "drug_name": drug_name,
        "measured_concentration": measured_concentration,
        "unit": unit,
        "target_min": min_conc,
        "target_max": max_conc,
        "status": status,
        "action": action,
        "recommendation": recommendation,
        "sampling_time": sampling_time,
    }


def check_drug_interactions(drug_name: str, concomitant_drugs: List[str]) -> List[Dict]:
    """
    检查药物相互作用

    参数:
        drug_name: 免疫抑制剂名称
        concomitant_drugs: 联用药物列表

    返回:
        相互作用列表
    """
    interactions_db = {
        "他克莫司": {
            "氟康唑": {
                "severity": "严重",
                "effect": "氟康唑显著抑制他克莫司的CYP3A4代谢，导致他克莫司血药浓度升高2-4倍",
                "recommendation": "合用时他克莫司剂量需降低50-70%，密切监测血药浓度",
            },
            "伏立康唑": {
                "severity": "严重",
                "effect": "伏立康唑为强效CYP3A4抑制剂，显著升高他克莫司血药浓度",
                "recommendation": "合用时他克莫司剂量需降低50-70%，密切监测血药浓度",
            },
            "伊曲康唑": {
                "severity": "严重",
                "effect": "伊曲康唑为CYP3A4抑制剂，可使他克莫司血药浓度显著升高",
                "recommendation": "合用时他克莫司剂量需降低50%，密切监测血药浓度",
            },
            "艾沙康唑": {
                "severity": "中等",
                "effect": "艾沙康唑为CYP3A4抑制剂，可能升高他克莫司血药浓度",
                "recommendation": "合用时监测他克莫司血药浓度，必要时调整剂量",
            },
            "奈玛特韦/利托那韦": {
                "severity": "严重",
                "effect": "利托那韦为强效CYP3A4抑制剂，显著升高他克莫司血药浓度",
                "recommendation": "合用时他克莫司剂量需降低60-80%，密切监测血药浓度",
            },
            "利福平": {
                "severity": "严重",
                "effect": "利福平诱导CYP3A4代谢，导致他克莫司血药浓度显著降低",
                "recommendation": "合用时他克莫司剂量可能需要增加2-3倍，监测血药浓度",
            },
            "五酯胶囊/软胶囊": {
                "severity": "中等",
                "effect": "五酯胶囊主要成分为五味子甲素，可抑制CYP3A4酶，显著升高他克莫司血药浓度",
                "recommendation": "合用时他克莫司剂量需降低30-50%，密切监测血药浓度",
            },
            "西柚汁": {
                "severity": "中等",
                "effect": "抑制肠道CYP3A4，显著增加他克莫司生物利用度",
                "recommendation": "避免服用他克莫司期间饮用西柚汁",
            },
        },
        "环孢素": {
            "氟康唑": {
                "severity": "严重",
                "effect": "氟康唑显著抑制CYP3A4代谢，环孢素血药浓度可升高2-3倍",
                "recommendation": "合用时环孢素剂量需降低50%，密切监测血药浓度",
            },
            "伏立康唑": {
                "severity": "严重",
                "effect": "伏立康唑为强效CYP3A4抑制剂，显著升高环孢素血药浓度",
                "recommendation": "合用时环孢素剂量需降低50%，密切监测血药浓度",
            },
            "伊曲康唑": {
                "severity": "严重",
                "effect": "伊曲康唑为CYP3A4抑制剂，可使环孢素血药浓度显著升高",
                "recommendation": "合用时环孢素剂量需降低50%，密切监测血药浓度",
            },
            "艾沙康唑": {
                "severity": "中等",
                "effect": "艾沙康唑为CYP3A4抑制剂，可能升高环孢素血药浓度",
                "recommendation": "合用时监测环孢素血药浓度，必要时调整剂量",
            },
            "奈玛特韦/利托那韦": {
                "severity": "严重",
                "effect": "利托那韦为强效CYP3A4抑制剂，显著升高环孢素血药浓度",
                "recommendation": "合用时环孢素剂量需降低60-80%，密切监测血药浓度",
            },
            "利福平": {
                "severity": "严重",
                "effect": "强效CYP3A4诱导剂，导致环孢素浓度显著降低",
                "recommendation": "避免合用，或环孢素剂量可能需要增加2-3倍",
            },
            "五酯胶囊/软胶囊": {
                "severity": "中等",
                "effect": "五酯胶囊可抑制CYP3A4酶，升高环孢素血药浓度",
                "recommendation": "合用时环孢素剂量需降低30-50%，密切监测血药浓度",
            },
            "西柚汁": {
                "severity": "中等",
                "effect": "抑制肠道CYP3A4，增加环孢素生物利用度",
                "recommendation": "避免服用环孢素期间饮用西柚汁",
            },
        },
        "霉酚酸酯": {
            "氟康唑": {
                "severity": "轻微",
                "effect": "氟康唑可能轻度影响霉酚酸酯代谢",
                "recommendation": "通常无需调整剂量，监测霉酚酸酯相关不良反应",
            },
            "伏立康唑": {
                "severity": "轻微",
                "effect": "伏立康唑可能轻度影响霉酚酸酯代谢",
                "recommendation": "通常无需调整剂量，监测霉酚酸酯相关不良反应",
            },
            "伊曲康唑": {
                "severity": "轻微",
                "effect": "伊曲康唑可能轻度影响霉酚酸酯代谢",
                "recommendation": "通常无需调整剂量，监测霉酚酸酯相关不良反应",
            },
            "艾沙康唑": {
                "severity": "轻微",
                "effect": "艾沙康唑可能轻度影响霉酚酸酯代谢",
                "recommendation": "通常无需调整剂量，监测霉酚酸酯相关不良反应",
            },
            "奈玛特韦/利托那韦": {
                "severity": "轻微",
                "effect": "利托那韦可能轻度影响霉酚酸酯代谢",
                "recommendation": "通常无需调整剂量，监测霉酚酸酯相关不良反应",
            },
            "利福平": {
                "severity": "中等",
                "effect": "利福平诱导UDP-葡萄糖醛酸转移酶，降低霉酚酸酯浓度",
                "recommendation": "监测疗效，可能需要增加剂量",
            },
            "五酯胶囊/软胶囊": {
                "severity": "轻微",
                "effect": "五酯胶囊对霉酚酸酯代谢影响较小",
                "recommendation": "通常无需调整剂量，监测疗效",
            },
            "西柚汁": {
                "severity": "轻微",
                "effect": "西柚汁对霉酚酸酯吸收影响较小",
                "recommendation": "通常无需调整剂量",
            },
        },
        "西罗莫司": {
            "氟康唑": {
                "severity": "严重",
                "effect": "氟康唑抑制CYP3A4代谢，显著升高西罗莫司血药浓度",
                "recommendation": "合用时西罗莫司剂量需降低50%，密切监测血药浓度",
            },
            "伏立康唑": {
                "severity": "严重",
                "effect": "伏立康唑为强效CYP3A4抑制剂，显著升高西罗莫司血药浓度",
                "recommendation": "合用时西罗莫司剂量需降低50-70%，密切监测血药浓度",
            },
            "伊曲康唑": {
                "severity": "严重",
                "effect": "伊曲康唑为CYP3A4抑制剂，显著升高西罗莫司血药浓度",
                "recommendation": "合用时西罗莫司剂量需降低50%，密切监测血药浓度",
            },
            "艾沙康唑": {
                "severity": "中等",
                "effect": "艾沙康唑为CYP3A4抑制剂，可能升高西罗莫司血药浓度",
                "recommendation": "合用时监测西罗莫司血药浓度，必要时调整剂量",
            },
            "奈玛特韦/利托那韦": {
                "severity": "严重",
                "effect": "利托那韦为强效CYP3A4抑制剂，显著升高西罗莫司血药浓度",
                "recommendation": "合用时西罗莫司剂量需降低60-80%，密切监测血药浓度",
            },
            "利福平": {
                "severity": "严重",
                "effect": "利福平诱导CYP3A4代谢，导致西罗莫司血药浓度显著降低",
                "recommendation": "合用时西罗莫司剂量可能需要增加2-3倍，监测血药浓度",
            },
            "五酯胶囊/软胶囊": {
                "severity": "中等",
                "effect": "五酯胶囊可抑制CYP3A4酶，升高西罗莫司血药浓度",
                "recommendation": "合用时西罗莫司剂量需降低30-50%，密切监测血药浓度",
            },
            "西柚汁": {
                "severity": "中等",
                "effect": "抑制肠道CYP3A4，增加西罗莫司生物利用度",
                "recommendation": "避免服用西罗莫司期间饮用西柚汁",
            },
        },
        "依维莫司": {
            "氟康唑": {
                "severity": "严重",
                "effect": "氟康唑抑制CYP3A4代谢，显著升高依维莫司血药浓度",
                "recommendation": "合用时依维莫司剂量需降低50%，密切监测血药浓度",
            },
            "伏立康唑": {
                "severity": "严重",
                "effect": "伏立康唑为强效CYP3A4抑制剂，显著升高依维莫司血药浓度",
                "recommendation": "合用时依维莫司剂量需降低50-70%，密切监测血药浓度",
            },
            "伊曲康唑": {
                "severity": "严重",
                "effect": "伊曲康唑为CYP3A4抑制剂，显著升高依维莫司血药浓度",
                "recommendation": "合用时依维莫司剂量需降低50%，密切监测血药浓度",
            },
            "艾沙康唑": {
                "severity": "中等",
                "effect": "艾沙康唑为CYP3A4抑制剂，可能升高依维莫司血药浓度",
                "recommendation": "合用时监测依维莫司血药浓度，必要时调整剂量",
            },
            "奈玛特韦/利托那韦": {
                "severity": "严重",
                "effect": "利托那韦为强效CYP3A4抑制剂，显著升高依维莫司血药浓度",
                "recommendation": "合用时依维莫司剂量需降低60-80%，密切监测血药浓度",
            },
            "利福平": {
                "severity": "严重",
                "effect": "利福平诱导CYP3A4代谢，导致依维莫司血药浓度显著降低",
                "recommendation": "合用时依维莫司剂量可能需要增加2-3倍，监测血药浓度",
            },
            "五酯胶囊/软胶囊": {
                "severity": "中等",
                "effect": "五酯胶囊可抑制CYP3A4酶，升高依维莫司血药浓度",
                "recommendation": "合用时依维莫司剂量需降低30-50%，密切监测血药浓度",
            },
            "西柚汁": {
                "severity": "中等",
                "effect": "抑制肠道CYP3A4，增加依维莫司生物利用度",
                "recommendation": "避免服用依维莫司期间饮用西柚汁",
            },
        },
    }

    results = []
    drug_interactions = interactions_db.get(drug_name, {})

    for concomitant_drug in concomitant_drugs:
        if concomitant_drug in drug_interactions:
            results.append({
                "concomitant_drug": concomitant_drug,
                **drug_interactions[concomitant_drug]
            })
        else:
            results.append({
                "concomitant_drug": concomitant_drug,
                "severity": "未发现",
                "effect": "暂无明确相互作用报道",
                "recommendation": "常规监测即可",
            })

    return results


def get_severity_color(severity: str) -> str:
    """
    根据相互作用严重程度返回对应颜色代码
    """
    color_map = {
        "严重": "#FF0000",
        "中等": "#FFA500",
        "轻微": "#FFFF00",
        "未发现": "#00FF00",
    }
    return color_map.get(severity, "#808080")


DOSE_RULES = {
    "他克莫司": {
        "min_dose_mg_per_kg": 0.05,
        "max_dose_mg_per_kg": 0.2,
        "dose_adjustment_threshold": 0.5,
    },
    "环孢素": {
        "min_dose_mg_per_kg": 2.0,
        "max_dose_mg_per_kg": 8.0,
        "dose_adjustment_threshold": 1.0,
    },
    "霉酚酸酯": {
        "min_dose_mg_per_kg": 20.0,
        "max_dose_mg_per_kg": 40.0,
        "dose_adjustment_threshold": 5.0,
    },
}


def validate_llm_dose_response(drug_name: str, suggested_dose: float, weight: float) -> Dict:
    """
    校验 LLM 给出的剂量建议是否在合理范围内

    参数:
        drug_name: 药物名称
        suggested_dose: LLM 建议的剂量 (mg/day)
        weight: 体重 (kg)

    返回:
        校验结果字典
    """
    rules = DOSE_RULES.get(drug_name)
    if not rules:
        return {"valid": True, "warning": "未知药物，无法校验"}

    dose_per_kg = suggested_dose / weight

    if dose_per_kg < rules["min_dose_mg_per_kg"]:
        return {
            "valid": False,
            "warning": f"建议剂量低于最低推荐剂量 {rules['min_dose_mg_per_kg']} mg/kg/天",
            "suggested_min": rules["min_dose_mg_per_kg"] * weight,
        }

    if dose_per_kg > rules["max_dose_mg_per_kg"]:
        return {
            "valid": False,
            "warning": f"建议剂量高于最高推荐剂量 {rules['max_dose_mg_per_kg']} mg/kg/天",
            "suggested_max": rules["max_dose_mg_per_kg"] * weight,
        }

    return {
        "valid": True,
        "dose_per_kg": round(dose_per_kg, 3),
        "within_range": True,
    }
