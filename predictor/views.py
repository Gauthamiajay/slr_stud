from django.shortcuts import render
from django.http import JsonResponse
from .models import train_model
import joblib  # To save/load models
import os

MODEL_PATH = "trained_model.pkl"

def home(request):
    return render(request, 'predictor/home.html')

def train_view(request):
    """Trains the model and saves it."""
    model, accuracy, error = train_model()
    if model:
        joblib.dump(model, MODEL_PATH)  # Save model
        return JsonResponse({"message": "Model trained successfully", "accuracy": accuracy, "error": error})
    return JsonResponse({"error": "Failed to train model"}, status=500)

def predict(request):
    """Handles form submission for making predictions."""
    if request.method == "POST":
        hours = request.POST.get('hours')
        try:
            hours = float(hours)
        except (ValueError, TypeError):
            return render(request, 'predictor/result.html', {"error": "Invalid input. 'hours' must be a number."})

        if not os.path.exists(MODEL_PATH):
            return render(request, 'predictor/result.html', {"error": "Model not trained yet. Please train the model first."})

        model = joblib.load(MODEL_PATH)
        prediction = model.predict([[hours]])[0]

        return render(request, 'predictor/result.html', {
            'hours': hours,
            'prediction': round(prediction, 2),
        })

    return render(request, 'predictor/home.html')

def predict_view(request):
    """Handles API requests for predictions."""
    hours = request.GET.get("hours")
    if hours is None:
        return JsonResponse({"error": "Missing 'hours' parameter"}, status=400)

    try:
        hours = float(hours)
    except ValueError:
        return JsonResponse({"error": "Invalid input. 'hours' must be a number."}, status=400)

    if not os.path.exists(MODEL_PATH):
        return JsonResponse({"error": "Model not trained yet. Please train the model first."}, status=500)

    model = joblib.load(MODEL_PATH)
    predicted_score = model.predict([[hours]])[0]

    return JsonResponse({"predicted_score": round(predicted_score, 2)})
