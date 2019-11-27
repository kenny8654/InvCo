<template lang='pug'>
#app
  .ui.visible.sidebar.left.inverted.vertical.menu
    h1 {{ title }}
    .ui.item Admin
    .ui.item Dashboard
    .ui.item Home
    .ui.item Statistic
  
  .pusher
    .ui.internally.celled.grid
      .row
        .four.wide.column
          h2 Number of recipe  :{{recipe}}
        .four.wide.column
          h2 Number of image  :{{image}}
        .six.wide.column
          h2 Empty
      .row 
        .fourteen.wide.column
          .ui.eight.stackable.cards
            .card
              .image
                img(src='https://i.imgur.com/yNwH3jV.jpg')
            .card
              .image
                img(src='https://i.imgur.com/yNwH3jV.jpg')
            .card
              .image
                img(src="https://i.imgur.com/U4eeRbY.png")
            

</template>


<script>
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
    title: "Inverse Cooking",
    recipe: 100,
    image: 100,
  }},

  methods: {
  
  }
}

</script>

<style lang="sass">
h1
  color : white 
</style>

