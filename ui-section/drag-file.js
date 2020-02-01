(function () {
    var holder = document.getElementById('drag-file');

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
            var para = document.createElement("option");
            var node = document.createTextNode(f.name + f.path);
            para.appendChild(node);
            var element = document.getElementById("path");
            element.appendChild(para);
        }
        
        return false;
    };
})();