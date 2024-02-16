from rest_framework.response import Response
from rest_framework.decorators import api_view
from api import nano_model
import pandas as pd

@api_view(['POST'])
def deliveryefficiencyinputDatapost(request):

    data = [{
        'Type': request.data.get('Type'),
        'MAT': request.data.get('MAT'),
        'TS': request.data.get('TS'),
        'CT': request.data.get('CT'),
        'TM': request.data.get('TM'),
        'Shape': request.data.get('Shape'),
        'Size': request.data.get('Size'),
        'Zeta Potential': request.data.get('ZetaPotential'),
        'Admin': request.data.get('Admin'),
    }]

    preds = nano_model.predict_df(pd.DataFrame(data))

    return Response(preds)