function clearErrors(){
    errors = document.getElementsByClassName('formerror');
    for( let item of errors)
    {
        item.innerHTMl=" ";
    }
}

function setError(id, error){
    element =document.getElementById(id);
    element.getElementsByClassName('formerror')[0].innerHTMl = error ;
}
 function validateform(){
    var returnval = true;
    clearErrors();
    var name = document.forms['myForm']["fAir"].value;
    if( name.legth<5){
        setError("name", "* Length is to short");
        return returnval= false
    }
 }