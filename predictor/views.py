from django.shortcuts import render
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
