const {ipcRenderer} = require('electron')
// let $ = require("jquery") 


const selectDirBtn = document.getElementById('select-directory')

selectDirBtn.addEventListener('click', (event) => {
  ipcRenderer.send('open-file-dialog')
})

ipcRenderer.on('selected-directory', (event, path) => {
  document.getElementById('selected-file').innerHTML = `${path}`
  

})