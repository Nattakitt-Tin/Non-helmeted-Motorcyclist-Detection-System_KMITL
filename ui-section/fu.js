var path_f = []
const {BrowserWindow,dialog,ipcMain} = require('electron').remote
const path = require('path')
const modalPath = path.join('file://', __dirname, 'get_po.html')
var vdo_h = 0
var pos = []
var file_p = []

function go() {

    var pyshell =  require('python-shell');
    var options = {
    args : [file_p] }

    pyshell.run('drawing.py',options,  function  (err, results)  {
     if  (err)  throw err;
    console.log(' finished.');
    // console.log(results)
    pos.push(results[0])
    pos.push(results[1])
    pos.push(results[2])
    pos.push(results[3])
    pos.push(results[4])
    // console.log('results', x1); 
    console.log(pos)
  
  });  



}

function open_win(w,h) {
    let win = new BrowserWindow({ width: w, height: h })
    win.loadURL(modalPath)
    win.show()
}  

function select_f() {

    const options = {
        title: 'Save an Image',
        filters: [
          { name: 'Movies', extensions: ['mkv', 'avi', 'mp4'] }
        ]
      }

    // dialog.showOpenDialog(options)

    ipcMain.on('open-file-dialog', (event) => {
        dialog.showOpenDialog(options, (files) => {
          if (files) {
            event.sender.send('selected-directory', files)
            // var replace = files.replace('\\','/')
            var x = document.createElement("VIDEO");
            x.setAttribute("id","pa")
            x.setAttribute("src",files)
            // var re = x.getAttribute("src")
            // var comre = re.replace("/\\/g","/")
            // console.log(comre)
            x.setAttribute("width", this.videoWidth);
            x.setAttribute("height", this.videoHeight);
            x.setAttribute("controls", "controls");
            document.body.appendChild(x);
            file_p.push(files)
            var d = document.getElementById('pa')
            d.addEventListener('loadedmetadata', function(e){
              console.log(d.videoWidth, d.videoHeight);
              // console.log(re)
              console.log(file_p)
              // open_win(d.videoWidth,d.videoHeight)
          });
    
          }
        })
      })
      
      
}

