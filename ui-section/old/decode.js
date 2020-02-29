var pyshell =  require('python-shell');

pyshell.on("message", function(message) {
    console.log(message)
  });
  
  setInterval(() => {
    pyshell.send(data);
  }, 300);