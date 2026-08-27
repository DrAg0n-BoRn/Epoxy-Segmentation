from dragon._helpers import wrap_text


def _format_model_name(raw_name: str, wrap_length: int = 12) -> str:
    name = ''
    if raw_name == "1_0-0_1":
        name = "PILF 0.10"
    elif raw_name == "1_0-0_05":
        name = "PILF 0.05"
    elif raw_name == "1_0-0_3":
        name = "PILF 0.30"
    elif raw_name == "1_0-0_5":
        name = "PILF 0.50"
    elif raw_name == "1_0-0_99":
        name = "PILF 0.99"
    elif raw_name == "GeneralizedDice-Focal":
        name = "Generalized Dice Focal"
    else:
        name = raw_name
    
    name = wrap_text(name, width=wrap_length)
    
    return name
