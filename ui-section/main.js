const {app, BrowserWindow,dialog} = require('electron')

function createWindow () {
    window = new BrowserWindow({width: 625, height: 640})
    window.loadFile('index.html')
    // window.setMenu(null)
}




app.on('ready', createWindow)

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
})

