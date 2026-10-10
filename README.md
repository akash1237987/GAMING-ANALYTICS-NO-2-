# Gaming Industry Analytics (Dynamic Version)

## Run locally
1. Install the packages: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`

## Demo login
- Username: `student`
- Password: `1234`

## Dynamic functionality
- Upload a CSV with game_name, platform, genre, year, sales, and rating columns.
- Add game records using a form.
- Edit/add/remove records in the data table and save changes.
- The analytics modules use the current active dataset.
- Download the current dataset as CSV.

Data modifications are held in Streamlit session state. Download your dataset if you need to keep changes after the app session ends. This is a coursework demo login, not production-grade authentication.
