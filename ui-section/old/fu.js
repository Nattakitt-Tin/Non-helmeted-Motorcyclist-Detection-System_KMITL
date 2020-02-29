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

