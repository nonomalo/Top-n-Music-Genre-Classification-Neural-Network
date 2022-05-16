from flask import Flask, render_template

app = Flask(__name__)


# route to music upload page
@app.route('/')
def index():
    return render_template('index.html')


# route to about project page
@app.route('/about')
def about():
    return render_template('about.html')


# route to team information page
@app.route('/team')
def team():
    return render_template('team.html')


# route to 404 error
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')


# route to server error
@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html')


if __name__ == '__main__':
    app.run(debug=True)
