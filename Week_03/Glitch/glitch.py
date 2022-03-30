from flask import Flask
import random

app = Flask(__name__)

topics=[
  {"id":1,"title":"html","body":"html is...."},
  {"id":2,"title":"css","body":"css is...."},
  {"id":3, "title":"js", "body":"js is ...."}
]

@app.route("/example")
def hello_world():
  return '<strong>random</strong> : '+str(random.random())
  

@app.route("/")
def home():
  return template('<h2>Welcome</h2> Hello, Web!')


def template(content):
  
  liTags=''
  for topic in topics:
    liTags=liTags+f'<li><a href="read/{topic["id"]}">{topic["title"]}</a></li>'
      
  return f'''
    <html>
    <body>
      <h1><a href="/">WEB</a></h1>
      <ol>
        {liTags}
      </ol>
      {content}
      
      <ul>
        <li><a href="/create/">create</a></li>
      </ul>
      
    </body>
    </html>
    '''
  
  
@app.route("/read/<int:id>/")
def read1(id):
  
    title=''
    body=''
    
    for topic in topics:
      if id==topic['id']:
        title=topic['title']
        body=topic['body']
        
    return f'''
    <html>
    <body>
      <h1><a href="/">WEB</a></h1>
      <ol>
        <li><a href="/read/1/">html</a></li>
        <li><a href="/read/2/">css</a></li>
        <li><a href="/read/3/">javascript</a></li>
      </ol>
      <h2>{title}</h2>
      Hello, {body}!
    </body>
    </html>
    '''

@app.route("/create/")
def create():
  content='''
    <form action="/create/">
    <p><input type="text" name="title" placeholder="title"></p>
    <p><textarea name="body" placeholder="body"></textarea></p>
    <p><input type="submit" value="create"></p>
    </form>
    '''
  return template(content)

@app.route("/update/")
def update():
  return 'update'

app.run()