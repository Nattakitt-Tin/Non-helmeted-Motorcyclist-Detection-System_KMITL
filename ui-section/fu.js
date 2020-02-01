var holder = document.getElementById('drag-file');
var path_f = []
const {BrowserWindow,dialog,ipcMain} = require('electron').remote
const path = require('path')
const modalPath = path.join('file://', __dirname, 'get_po.html')
var vdo_h = 0
var pos = []
  

holder.ondragover = () => {
    return false;
};

holder.ondragleave = () => {
    return false;
};

holder.ondragend = () => {
    return false;
};

holder.ondrop = (e) => {
    e.preventDefault();

    for (let f of e.dataTransfer.files) {
        console.log('File(s) you dragged here: ', f.path)
        var list_f = document.createElement('option');
        list_f.textContent = f.path;
        document.getElementById("l_file").appendChild(list_f)
        path_f.push(f.path)         
    }
    console.log(path_f)
    return false;
};


function go() {

    var pyshell =  require('python-shell');
    var options = {
    args : [path_f] }

    pyshell.run('drawing.py',options,  function  (err, results)  {
     if  (err)  throw err;
    console.log('hello.py finished.');
    pos.push(results[0])
    pos.push(results[1])
    pos.push(results[2])
    pos.push(results[3])
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
            var x = document.createElement("VIDEO");
            x.setAttribute("id","pa")
            x.setAttribute("src",files)
            x.setAttribute("width", this.videoWidth);
            x.setAttribute("height", this.videoHeight);
            x.setAttribute("controls", "controls");
            document.body.appendChild(x);
            var d = document.getElementById('pa')
            d.addEventListener('loadedmetadata', function(e){
              console.log(d.videoWidth, d.videoHeight);
              // open_win(d.videoWidth,d.videoHeight)
          });
    
          }
        })
      })
      
      
}

