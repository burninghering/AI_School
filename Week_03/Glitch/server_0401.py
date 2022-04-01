from flask import Flask, request, redirect
import sqlite3

app = Flask(__name__)

topics = [
  {"id":1, "title":"html", "body":"html is ...."},
  {"id":2, "title":"css", "body":"css is ...."},
  {"id":3, "title":"js", "body":"js is ...."}
]


def template(content, id=None):
  contextUI = ''
  if id != None:
    contextUI = '<input type="submit" value="delete" class="btn btn-dark">'
  conn = sqlite3.connect('db.sqlite3')
  cs = conn.cursor()
  cs.execute('SELECT * FROM topics')
  topics = cs.fetchall()
  conn.close()
  liTags = ''
  for topic in topics:
    liTags = liTags + f'<li><a href="/read/{topic[0]}/">{topic[1]}</a></li>'
  return f'''
  <html>
    <head>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
      <style>
        h1{{
          border-bottom:10px solid green;
        }}
      </style>
    </head>
    <body class="container">
      
      <input type="button" value="night" onclick="
      document.querySelector('body').style.backgroundColor='black';
      document.querySelector('body').style.color='white';
      ">
      
      <input type="button" value="day" onclick="
      document.querySelector('body').style.backgroundColor='white';
      document.querySelector('body').style.color='black';
      ">
      
      <h1><a href="/">WEB</a></h1>
      <ol>
        {liTags}
      </ol>
      {content}
      <form action="/delete/{id}/" method="POST">
        <div class="btn-group" role="group" aria-label="Basic example">  
          <a href="/create/" class="btn btn-dark">create</a>
          <input type="submit" value="delete" class="btn btn-dark">
        </div>
      </form>
      
      
      
      <!-- Copy and Paste Me -->
<div class="glitch-embed-wrap" style="height: 420px; width: 100%;">
  <iframe
    src="https://glitch.com/embed/#!/embed/flask---daegu-ai-school?path=server.py&previewSize=0"
    title="flask---daegu-ai-school on Glitch"
    allow="geolocation; microphone; camera; midi; encrypted-media; xr-spatial-tracking; fullscreen"
    allowFullScreen
    style="height: 100%; width: 100%; border: 0;">
  </iframe>
</div>
      
      
      
    </body>
  </html>
  '''

@app.route("/")
def index():
  return template('<h2>Welcome</h2>Hello, WEB!')

@app.route("/read/<int:id>/")
def read(id):
  conn = sqlite3.connect('db.sqlite3') #접속
  cs = conn.cursor() #커서를 얻어내고
  cs.execute('SELECT * FROM topics WHERE id=?', (id,)) #id의 값을 입력해서 1건 가져온다
  topic=cs.fetchone() #한 건을 fetchone으로 가져온다
  conn.close()
  
  
  title = topic[1]
  body = topic[2]
  return template(f'<h2>{title}</h2>{body}', id)


@app.route('/create/')
def create():
  content = '''
    <form action="/create_process/" method="POST">
      <p><input type="text" name="title" placeholder="title"></p>
      <p><textarea name="body" placeholder="body"></textarea></p>
      <p><input type="submit" value="create"></p>
    </form>
  '''
  return template(content)

@app.route('/create_process/', methods=['POST'])
def create_process():
  title=request.form['title'] #타이틀과 바디값을 브라우저에서 받고,
  body=request.form['body']
  conn=sqlite3.connet('db.sqlite3')
  cs=conn.cursor()
  cs.execute('INSERT INTO topics(title,body) VALUES(?,?)',(title,body))
  id=cs.lastrowid #db의 마지막 id를 받아낸다
  conn.commit() #커밋 실행하기 전엔, db에 쓰이기 전이고 commit을 실행해야 db에 쓰인다
  conn.close()

  return redirect(f'/read/{id}/') #nextId-1 을 나타낼 필요없이, db가 처리해준 id를 넣으면 된다.


@app.route('/delete/<int:id>/', methods=['POST'])
def delete(id):
  conn = sqlite3.connect('db.sqlite3') 
  cs = conn.cursor() 
  cs.execute('DELETE FROM topics WHERE id=?', (id,))
  conn.commit() 
  conn.close()
  
  return redirect('/')

  
  
# # @app.route('/update/')
# # def update():
# #   return 'Update'
 

app.run()

