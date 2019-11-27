import Vue from 'vue'

import App from './components/App.vue'
import Upload from './components/Upload.vue'

if(document.getElementById('app')){
  new Vue({
    el: '#app',
    render: h => h(App),
  })
}

else if(document.getElementById('upload')){
  new Vue({
    el: '#upload',
    render: h => h(Upload),
  })
}
