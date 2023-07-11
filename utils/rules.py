# --------------- Condition class --------------
from enum import Enum
import re


# confidence_threshold = 101 #For manual verification of all docs
confidence_threshold = 95  # cutoff for automatic verification
rules = [
    {
        "description": f"ADDRESSEE confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESSEE",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ADDRESS_LINE_1 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESS_LINE_1",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ADDRESS_LINE_2 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESS_LINE_2",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"CITY confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "CITY",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"STATE confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "STATE",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ZIP_CODE_4 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ZIP_CODE_4",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"REGULATORY_APPROVAL_ID confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "REGULATORY_APPROVAL_ID",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
]


class Condition:
    _data = None
    _conditions = None
    _result = None

    def __init__(self, data, conditions):
        self._data = data
        self._conditions = conditions

    def check(self, field_name, obj):
        r, s = [], []
        for c in self._conditions:
            # Matching field_name or field_name_regex
            condition_setting = c.get("condition_setting")
            if c["field_name"] == field_name or (
                c.get("field_name") is None
                and c.get("field_name_regex") is not None
                and re.search(c.get("field_name_regex"), field_name)
            ):
                field_value, block = None, None
                if obj is not None:
                    field_value = obj.get("value")
                    block = obj.get("block")
                    confidence = obj.get("confidence")

                if c["condition_type"] == "Required" and (
                    obj is None or field_value is None or len(str(field_value)) == 0
                ):
                    r.append(
                        {
                            "message": f"The required field [{field_name}] is missing.",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )
                elif (
                    c["condition_type"] == "ConfidenceThreshold"
                    and obj is not None
                    and c["condition_setting"] is not None
                    and float(confidence) < float(c["condition_setting"])
                ):
                    r.append(
                        {
                            "message": f"The field [{field_name}] confidence score {confidence} is LOWER than the threshold {c['condition_setting']}",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )

                elif (
                    field_value is not None
                    and c["condition_type"] == "ValueRegex"
                    and condition_setting is not None
                    and re.search(condition_setting, str(field_value)) is None
                ):
                    r.append(
                        {
                            "message": f"{c['description']}",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )

                # field has condition defined and sastified
                s.append(
                    {
                        "message": f"The field [{field_name}] confidence score is {confidence}.",
                        "field_name": field_name,
                        "field_value": field_value,
                        "condition_type": str(c["condition_type"]),
                        "condition_setting": condition_setting,
                        "condition_category": c["condition_category"],
                        "block": block,
                    }
                )

        return r, s

    def check_all(self):
        if self._data is None or self._conditions is None:
            return None
        # if rule missed, this list holds list
        broken_conditions = []
        rules_checked = []
        broken_conditions_with_all_fields_displayed = []
        # iterate through rules_data dictionary
        # key is a field, value/obj is a
        # dictionary, for instance name_rule_data
        # is the dictionary for ADDRESSEE
        for key, obj in self._data.items():
            value = None
            if obj is not None:
                value = obj.get("value")

            if value is not None and type(value) == str:
                value = value.replace(" ", "")
            # Check if this field passed rules:
            # if so, r and s are both []
            # if not, r is list of one or more of the dictionaries
            # seen in the check function
            r, s = self.check(key, obj)
            # If field missed rule
            if r and len(r) > 0:
                # append to bc list
                broken_conditions += r
            # If rule checked for field
            if s and len(s) > 0:
                rules_checked += s
            # If field missed or not
            # append to b_c_w_a_f_d
            if s and len(s) > 0:
                if r and len(r) > 0:
                    # If rule missed on this field, record it
                    broken_conditions_with_all_fields_displayed += r
                else:
                    # If no rule missed, record field
                    broken_conditions_with_all_fields_displayed += s

        # apply index
        idx = 0
        # iterate through dictionaries
        for r in broken_conditions_with_all_fields_displayed:
            idx += 1
            r["index"] = idx
        # If at least one rule missed, display with it all fields
        # otherwise broken c
        if broken_conditions:
            broken_conditions = broken_conditions_with_all_fields_displayed
        return broken_conditions, rules_checked


def get_missed_and_checked_rules(reg_id_conf, name_conf, centene_format, rules):
    name_rule_data = {
        "value": centene_format["ADDRESSEE"],
        "confidence": name_conf,
        "block": None,
    }
    ad1_rule_data = {
        "value": centene_format["ADDRESS_LINE_1"],
        "confidence": name_conf,
        "block": None,
    }
    ad2_rule_data = {
        "value": centene_format["ADDRESS_LINE_2"],
        "confidence": name_conf,
        "block": None,
    }
    city_rule_data = {
        "value": centene_format["CITY"],
        "confidence": name_conf,
        "block": None,
    }
    state_rule_data = {
        "value": centene_format["STATE"],
        "confidence": name_conf,
        "block": None,
    }
    zip_rule_data = {
        "value": centene_format["ZIP_CODE_4"],
        "confidence": name_conf,
        "block": None,
    }
    # reg_id moved to bottom to match where it is found on page
    # labelers wanted the webpage they see to display regid as the last field
    reg_id_rule_data = {
        "value": centene_format["REGULATORY_APPROVAL_ID"],
        "confidence": reg_id_conf,
        "block": None,
    }
    rules_data = {
        "ADDRESSEE": name_rule_data,
        "ADDRESS_LINE_1": ad1_rule_data,
        "ADDRESS_LINE_2": ad2_rule_data,
        "CITY": city_rule_data,
        "STATE": state_rule_data,
        "ZIP_CODE_4": zip_rule_data,
        "REGULATORY_APPROVAL_ID": reg_id_rule_data,
    }

    # Validate business rules
    con = Condition(rules_data, rules)
    rule_missed, rule_checked = con.check_all()
    return rule_missed,rule_checked





