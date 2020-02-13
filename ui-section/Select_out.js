
function se(){
const fs = require("fs");
const {dialog} = require("electron").remote;
  dialog.showOpenDialog({
    title:"Select a folder",
    properties: ["openDirectory"]
}, (folderPaths) => {    
        console.log(folderPaths);
});
  
}
