from django.shortcuts import render
from django.http import JsonResponse
from .models import train_model

def home(request):
    return render(request, 'predictor/home.html')

def predict(request):
    if request.method == "POST":
        hours = float(request.POST['hours'])
        model, accuracy, error = train_model()
        prediction = model.predict([[hours]])[0]

        return render(request, 'predictor/result.html', {
            'hours': hours,
            'prediction': round(prediction, 2),
            'accuracy': round(accuracy, 2),
            'error': round(error, 2)
        })
    return render(request, 'predictor/home.html')
def train_view(request):
    model, accuracy, error = train_model()
    if model:
        return JsonResponse({"message": "Model trained successfully", "accuracy": accuracy, "error": error})
    return JsonResponse({"error": "Failed to train model"}, status=500)
def predict_view(request):
    hours = request.GET.get("hours")
    if hours is None:
        return JsonResponse({"error": "Missing 'hours' parameter"}, status=400)

    try:
        hours = float(hours)
        predicted_score = predict_score([hours])
        return JsonResponse({"predicted_score": predicted_score.tolist()})
    except ValueError:
        return JsonResponse({"error": "Invalid input. 'hours' must be a number."}, status=400)