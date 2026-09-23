import pandas as pd


def predict_patient(
    patient,
    model,
    scaler,
    explainer
):

    # ==========================================
    # CREATE DATAFRAME
    # ==========================================

    data = pd.DataFrame({

        "Age": [patient.Age],

        "BP": [patient.BP],

        "Creatinine": [patient.Creatinine]

    })


    # ==========================================
    # SCALE DATA
    # ==========================================

    scaled_data = scaler.transform(
        data
    )


    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(
        scaled_data
    )


    # ==========================================
    # PROBABILITY
    # ==========================================

    probability = model.predict_proba(
        scaled_data
    )


    # ==========================================
    # SHAP
    # ==========================================

    shap_values = explainer.shap_values(
        scaled_data
    )


    print("SHAP TYPE:", type(shap_values))

    if hasattr(shap_values, "shape"):
        print("SHAP SHAPE:", shap_values.shape)


    # ==========================================
    # HANDLE SHAP OUTPUT
    # ==========================================

    if isinstance(shap_values, list):

        # Older SHAP format
        #
        # [
        #   class_0_values,
        #   class_1_values
        # ]

        shap_values_ckd = shap_values[1][0]


    elif shap_values.ndim == 3:

        # Example:
        # (1, 3, 2)
        #
        # 1 = patient
        # 3 = features
        # 2 = classes

        shap_values_ckd = shap_values[0, :, 1]


    elif shap_values.ndim == 2:

        # Example:
        # (1, 3)
        #
        # Already one patient's
        # feature contributions

        shap_values_ckd = shap_values[0]


    else:

        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )


    # ==========================================
    # CHECK FEATURE COUNT
    # ==========================================

    if len(shap_values_ckd) != 3:

        raise ValueError(
            f"Expected 3 SHAP values, "
            f"but got {len(shap_values_ckd)}"
        )


    # ==========================================
    # EXPLANATION
    # ==========================================

    explanation = {

        "Age": float(
            shap_values_ckd[0]
        ),

        "BP": float(
            shap_values_ckd[1]
        ),

        "Creatinine": float(
            shap_values_ckd[2]
        )

    }


    # ==========================================
    # RETURN RESULT
    # ==========================================

    return {

        "prediction": int(
            prediction[0]
        ),

        "probability": float(
            probability[0][1]
        ),

        "explanation": explanation

    }