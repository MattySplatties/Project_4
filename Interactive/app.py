## pip install Flask

from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

def get_locations(city, biz_name):
    url = "https://api.yelp.com/v3/businesses/search?location=" + city + "&term=" + biz_name + "&sort_by=best_match&limit=3"

    headers = {
        "accept": "application/json",
        "Authorization": yelp_api_key
    }

    response = requests.get(url, headers=headers).json()
    res = response['businesses']
    store = []
    loc = []
    biz_id_list = []
    geo_lat = []
    geo_lng = []
    rating = []
    for biz in res:
        store.append(biz['name'])
        loc.append(biz['location']['address1'])
        biz_id_list.append(biz['id'])
        geo_lat.append(round(biz['coordinates']['latitude'],3))
        geo_lng.append(round(biz['coordinates']['longitude'],3))   
        
    selection_prompt = "Enter store number you wish to choose:"
    for i in range(0,3):
        location = f"{i}:&nbsp;{store[i]}&nbsp;{loc[i]}"        
        selection_prompt += location

    render_template("location_selector.html", prompt=selection_prompt)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        city = request.form['city']
        biz_name = request.form['biz_name']
        get_locations(city, biz_name)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)