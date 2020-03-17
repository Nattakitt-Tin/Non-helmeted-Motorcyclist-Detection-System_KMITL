const {app, BrowserWindow,dialog} = require('electron')

function createWindow () {
    window = new BrowserWindow({width: 680, height: 400,icon:"logo2.png"})
    window.loadFile('index.html')
    
    // window.setMenu(null)
}



app.on('ready', createWindow)

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
})

