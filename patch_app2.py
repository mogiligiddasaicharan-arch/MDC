path = "app.py"
content = open(path, encoding="utf-8").read()

old = """        CONFIDENCE_THRESHOLD = 0.75
        low_confidence = result["domain_confidence"] < CONFIDENCE_THRESHOLD
        response = {
            "domain": result["domain"],
            "domain_confidence": round(result["domain_confidence"], 4),
            "defect": result["defect"],
            "defect_confidence": round(result["defect_confidence"], 4),
            "domain_probabilities": {k: round(v, 4) for k, v in result["domain_probabilities"].items()},
            "defect_probabilities": {k: round(v, 4) for k, v in result["defect_probabilities"].items()},
            "gradcam_base64": gradcam_b64,
            "low_confidence_warning": low_confidence,
            "message": "Low confidence: image may not match a trained domain" if low_confidence else "OK",
        }"""

new = """        CONFIDENCE_THRESHOLD = 0.75
        low_confidence = result["domain_confidence"] < CONFIDENCE_THRESHOLD

        if low_confidence:
            defect_out = "N/A (domain uncertain)"
            defect_conf_out = 0
            defect_probs_out = {}
        else:
            defect_out = result["defect"]
            defect_conf_out = round(result["defect_confidence"], 4)
            defect_probs_out = {k: round(v, 4) for k, v in result["defect_probabilities"].items()}

        response = {
            "domain": result["domain"],
            "domain_confidence": round(result["domain_confidence"], 4),
            "defect": defect_out,
            "defect_confidence": defect_conf_out,
            "domain_probabilities": {k: round(v, 4) for k, v in result["domain_probabilities"].items()},
            "defect_probabilities": defect_probs_out,
            "gradcam_base64": gradcam_b64,
            "low_confidence_warning": low_confidence,
            "message": "Low confidence: image may not match a trained domain. Defect prediction suppressed." if low_confidence else "OK",
        }"""

if old not in content:
    print("PATTERN NOT FOUND - app.py differs from expected, no changes made.")
else:
    content = content.replace(old, new)
    open(path, "w", encoding="utf-8").write(content)
    print("app.py patched: defect prediction now suppressed when domain confidence is low.")
