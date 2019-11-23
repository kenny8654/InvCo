<template lang='pug'>
#app
  //- h1 hello {{ title }}
  .bg
    form(action='' method='post' enctype="multipart/form-data")
      div.ui.action.input.file-upload
       input(type='file' ref='file' @change='handleFileUpload()' accept='image/*' name='img')
       button.ui.olive.button(@click.prevent='submitFile()' type='submit') Upload
  // .ui.unstackable.items(v-if='file.length > 0')
  //   .item
  //     .image
  //       img(:src='file')
  //     .content
  //       a.header Header
  //       .meta
  //         span Description
  //       .description
  //         p IIII


</template>


<script>
import axios from 'axios'
import 'semantic-ui-offline/semantic.min.css'
import 'jquery'
import $ from 'jquery'


export default {

  beforeDestory() {
    document.removeEventListener('keyup', this.key)
  },

  created() {
    document.addEventListener('keyup', this.key)
  },

  data() { return {
    title: "Sunny Test",
    selectedFile: null,
    imageData: '',
    file:'',
    uploadPercentage: 0

  }},

  methods: {
    
    onImgSelected(event){
      var input = event.target;
      this.selectedFile = event.target.files[0]
      if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = (e) => {
          this.imageData = e.target.result;
        }
        reader.readAsDataURL(input.files[0]);
      }
    },

    submitFile(){
      let formData = new FormData();
      formData.append('image', this.file); //required
      console.log(formData.get('image'))
      
      axios.post('/upload/', formData, {
        headers: {'X-CSRFToken': this.getCookie('csrftoken')}, 
        onUploadProgress: function( progressEvent ) {
          this.uploadPercentage = parseInt( Math.round( ( progressEvent.loaded * 100 ) / progressEvent.total ) );
        }.bind(this)})
        .then(response => {
        console.log(response)
        })
  },
  handleFileUpload(event){
    // this.imageData = event.target.files[0] 
    this.file = this.$refs.file.files[0]
  },
  getCookie (name) {
    var value = '; ' + document.cookie
    var parts = value.split('; ' + name + '=')
    if (parts.length === 2) return parts.pop().split(';').shift()
  },

}
}

</script>
<style lang="sass">
.file-upload
  width: 600px
  margin: 0 auto
  padding: 20px
  left: 430px
  top: 455px
  z-index: 10px

body
  background-image: url(https://images.unsplash.com/photo-1472393365320-db77a5abbecc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2850&q=80)
  background-size: cover
  background-position: center
  height: 100vh
  width: 100vw  

</style>
