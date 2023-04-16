
$(document).ready(function(){
    $(window).scroll(function(){
        if ($(this).scrollTop() > 100) {
            $('.back-to-top').fadeIn();
        } else {
            $('.back-to-top').fadeOut();
        }
    });

    $('.back-to-top').click(function(){
        $('html, body').animate({scrollTop : 0},800);
        return false;
    });
});










var newsList = document.querySelector('.news-list');
window.addEventListener("load", function(e){
    e.preventDefault();
    const apiKey = 'UhEM6sCXRqA_ge-gfOiEXzAOAODhv9kB9WbFqk1clDg';
    let q = "agriculture+AND+(crop+OR+livestock+OR+farming)";
    let url = `https://api.newscatcherapi.com/v2/search?q=${q}&countries=IN&`;
    fetch(url, {
        method: 'GET',
        headers: {
            'x-api-key':apiKey
        }
    }).then((res)=>{
        return res.json()
    }).then((data)=>{
        console.log(data.articles);
        data.articles.forEach(article => {
            let div1 = document.createElement('div');
            div1.classList.add('news-card');
            let div2 = document.createElement('div');
            div2.classList.add('meta');
            let div3 = document.createElement('div');
            div3.classList.add('img');
            div3.style.backgroundImage = `url(${article.media})`;
            if (div3.style.backgroundImage == null){
                div3.style.backgroundImage = "url(./src/2.png)";
            }
            let detailsDiv = document.createElement('div');
            detailsDiv.classList.add('details');
            let descDiv = document.createElement('div');
            descDiv.classList.add('description');
            let h2 = document.createElement('h2');
            h2.innerText = article.title;
            let p = document.createElement('p');
            p.innerText = article.excerpt;
            let readMore = document.createElement('p');
            p.classList.add('read-more');
            let a = document.createElement('a');
            a.setAttribute('href',article.link);
            a.setAttribute('target','_blank');
            a.innerText = "Read More";

            div2.appendChild(div3);
            div2.appendChild(detailsDiv);
            readMore.appendChild(a);
            descDiv.appendChild(h2);
            descDiv.appendChild(p);
            descDiv.appendChild(readMore);
            div1.appendChild(div2);
            div1.appendChild(descDiv);
            newsList.appendChild(div1);

        });
    }).catch((err)=>{
        console.log(err);
    });
});
