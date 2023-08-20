import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecogrealtime-bb854-default-rtdb.firebaseio.com/"
})

ref=db.reference('Victims')

data={
    "1003":
        {
            "name":"Dhivya",
            "age":20,
            "sex":"female",
            "height":160,
            "weight":50,
            "state":"Tamilnadu",
            "city":'Cuddalore'
        },
    "1004":
        {
            "name":"Harshini",
            "age":21,
            "sex":"female",
            "height":130,
            "weight":30,
            "state":"Tamilnadu",
            "city":'Tenkasi'
        },
    "1005":
    {
        "name":"Indhu",
            "age":20,
            "sex":"female",
            "height":130,
            "weight":30,
            "state":"Tamilnadu",
            "city":'Karur'
    }
}

for key,value in data.items():
    ref.child(key).set(value)

print("Data added to DATABASE")