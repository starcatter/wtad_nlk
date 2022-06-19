# usage

## init text data
`python3 transator.py prepare in/pol-eng.zip pol.txt pl-en`

or unpack data and call

`python3 transator.py prepare in/pol-eng/pol.txt pl-en`

## build model
`python3 transator.py create pl-en pl-en-trans`

## run model
`python3 transator.py run pl-en pl-en-trans`