# Q2: 1.7.2

# Q3: 0.533

import pickle

with open('pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

lead = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

X = dv.transform([lead])

y_pred = model.predict_proba(X)[0, 1]

print(round(y_pred, 3)) # 0.533

# Q4: {"conversion_probability":0.769799,"conversion":true} --> 0.770

# Q5: 3.11.15-slim-bookworm

# Q6: {"conversion_probability":0.769799,"conversion":true} --> 0.770