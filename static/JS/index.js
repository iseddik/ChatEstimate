function clearInputFields() {
    var input1 = document.getElementById('name');
    input1.value = '';
}

const scrollingElement = document.getElementById("scrollable-container");

const config = { childList: true };

const callback = function (mutationsList, observer) {
    for (let mutation of mutationsList) {
        if (mutation.type === "childList") {
            window.scrollTo(0, document.body.scrollHeight);
        }
    }
};
