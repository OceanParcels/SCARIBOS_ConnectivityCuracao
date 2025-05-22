'''
Combine figures for manuscript 1
'''

#%%
import fitz

# meri or surface:
type = 'meri'

def combine_pdfs_on_one_page(pdf_files, output_file):

    width  = 500 
    height = 530 

    pdf_document  = fitz.open()
    combined_page = pdf_document.new_page(width=width, height=height)
    positions     = [(0, 0.1), (0, height / 1.95)]

    for i, pdf_file in enumerate(pdf_files):
        pdf         = fitz.open(pdf_file)
        source_page = pdf.load_page(0)
        if i == 0:
            scaling_factor = 2.5
        else:
            scaling_factor = 1
        rect        = source_page.rect
        scaled_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width * scaling_factor, rect.y0 + rect.height * scaling_factor)
        combined_page.show_pdf_page(
            fitz.Rect(positions[i][0], positions[i][1], positions[i][0] + width, positions[i][1] + height / 2),
            pdf,
            0,
            clip=scaled_rect
        )
        pdf.close()

    pdf_document.save(output_file)
    pdf_document.close()

combine_pdfs_on_one_page(
    [f"SCARIBOS_V8_avg_{type}_ALLYEARS_HQ.pdf", f"SCARIBOS_V8_avg_{type}_MONTHLY.pdf"],
    f"SCARIBOS_V8_avg_{type}_PAPER.pdf"
)



# %%
